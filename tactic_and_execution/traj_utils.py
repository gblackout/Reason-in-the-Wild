from generate import MyLlamaTokenizer
from prompts import tactic_exec_prompt
from .thought_action_observation_data_structure import Trajectory, Response, Feedback, ExpTuple
from copy import deepcopy
from utils import all_exists
from .consts import PYTHON_INTERPRETER, EXEC_OK, EXEC_SOLVED
from typing import Union, Optional, Dict
from collections import defaultdict


def make_init_prompt(sample: Union[str, Dict], tactic: str, sample_input_key: Optional[str] = None):
    if isinstance(sample, str):
        sample_input = sample
    else:
        sample_input = sample[sample_input_key] if all_exists(sample_input_key) else sample['input']
    return tactic_exec_prompt.format(**{
        'question-input': sample_input,
        'tactic-input': tactic
    })


def manual_action(messages, t, an, ain, aout):
    messages.append({'role': 'assistant', 'content': '\n'.join([
        '### Thought',
        f"{t}",
        '### Action',
        '## Name',
        f"{an}",
        '## Input',
        f"{ain}",
        '## Output',
        f"{aout}"
    ])})
    return messages


def manual_feedback(traj, content, exp_ind=None, observer='user', feedback_status='feedback wrong'):
    if all_exists(exp_ind):
        pass
    else:
        exp_ind = -1
    exp = traj.exp_ls[exp_ind]
    fb = Feedback(
        observer=observer,
        content=content,
        feedback_status=feedback_status,
    )
    if exp.observation is None:
        exp.observation = [fb]
    else:
        exp.observation.append(fb)


def get_program(
        traj: Trajectory,
        program_action_name: str,
        return_exp: bool = True,
        no_prog_ok: bool = False,
        verbose: bool = True
):

    prog_exps = [exp for exp in traj.exp_ls if exp.response.action_name == program_action_name]

    if len(prog_exps) > 1:
        if verbose:
            print('warning multiple write program resps detected, returning the first one')
        prog_exp = deepcopy(prog_exps[0])

    elif len(prog_exps) == 0:
        if verbose:
            print('no prog resp found')
        if no_prog_ok:
            prog_exp = None
        else:
            return None
    else:
        prog_exp = deepcopy(prog_exps[0])

    # find the last revise code if there exists one
    revise_exps = [exp for exp in traj.exp_ls if exp.response.action_name == 'Revise code']
    if len(revise_exps) == 0:
        pass
    else:
        revise_exp = deepcopy(revise_exps[-1])
        prog_exp = revise_exp

    prog_exp = prog_exp if return_exp else (prog_exp.response.action_output if all_exists(prog_exp) else None)

    return prog_exp


def find_exp(
        traj: Trajectory,
        action_name: str,
        after: Union[None, ExpTuple, int] = None,
        before: Union[None, ExpTuple, int] = None,
        exec_status:  Optional[str] = None,
        which_one: str = 'first'
):
    """
        find the first/last exp from traj with action_name and exec_status that is before and/or after some ExpTuples

    :param traj:
    :param action_name:
    :param after:
    :param before:
    :param exec_status:
    :param which_one:
        give me either "first", "last"
    :return:
    """
    if traj.exp_ls is None:
        return None, None

    def find_ind(tgt):
        if isinstance(tgt, int):
            ind = tgt
            if ind >= len(traj.exp_ls):
                raise ValueError(f'{ind} greater than len(exp_ls) = {len(traj.exp_ls)}')
        elif isinstance(tgt, ExpTuple):
            inds = [ind for ind, e in enumerate(traj.exp_ls) if e is tgt]
            assert len(inds) <= 1, 'somehow this traj has ExpTuple refs pointing to the same object'
            assert len(inds) == 1, 'ExpTuple not exists in exp_ls'
            ind = inds[0]
        else:
            raise ValueError('before/after can only be int or ExpTuple')
        return ind

    lind, rind = 0, len(traj.exp_ls)
    if all_exists(after):
        lind = find_ind(after) + 1 # skip the found one
    if all_exists(before):
        rind = find_ind(before)

    res_exps = [(ind, exp) for ind, exp in enumerate(traj.exp_ls)][lind:rind]
    if len(res_exps) == 0:
        return None, None

    match_exps = []
    for ind, exp in res_exps:
        atype = exp.action_type
        if all_exists(atype) and atype.action_name == action_name:
            if all_exists(exec_status):
                if exp.exec_status == exec_status:
                    match_exps.append((ind, exp))
            else:
                match_exps.append((ind, exp))

    if len(match_exps) == 0:
        return None, None

    if which_one == 'first':
        return match_exps[0]
    elif which_one == 'last':
        return match_exps[-1]
    else:
        raise ValueError(f'which_one can only be "first" or "last", but {which_one} given')


def make_auto_filter_question_prompt(sample, program_action_name, traj_key='traj', remove_comments=True):
    """
        turn a sample w/ traj into <question, program, obs> form, this is used for building icl prompts for trivial
        program detection

    :param sample:
    :return:
    """

    traj = Trajectory.from_json(sample[traj_key])
    prog_exp = get_program(traj, program_action_name=program_action_name)

    if prog_exp is None:
        return None

    prog = prog_exp.response.action_output
    if prog is None:
        return None

    if remove_comments:
        parts = prog.split('\n')
        nocomment_parts = [e.split('#')[0] for e in parts]
        nocomment_parts = [e for e in nocomment_parts if (e != '') and (not e.isspace())]
        prog = '\n'.join(nocomment_parts)

    # collect all outputs, i.e., feedbacks with observer being PYTHON_INTERPRETER
    obs = ('\n\n'.join(str(fb) for fb in prog_exp.observation if fb.observer == PYTHON_INTERPRETER)) if (
        all_exists(prog_exp.observation)) else ''

    example_prompt = (
        '=== Question and answer\n\n'
        f"{sample['input']}\n\nAnswer: {sample['label']}\n\n"
        '=== Proposed program\n\n'
        f'{prog}\n\n'
        '=== outputs\n\n'
        f'{obs}'
    )

    return example_prompt


def make_one_icl_prompt(sample, traj_key='traj', program_action_name='Write program', include_expected_answer=False):
    """
    question + plan (with HINTS removed) + first write program t + last revise code a
    NOTE this is ad-hoc because t may be wrong and not match with a, but we can try this first

    :param sample:
    :return:
    """

    traj = Trajectory.from_json(sample[traj_key])

    plan_resp = deepcopy(traj.exp_ls[0].response)
    assert plan_resp.action_name == 'Plan', \
        f'for icl example traj, the first step must be Plan, but {plan_resp.action_name} is given'

    # remove hints from plan and make plan str
    output_lines = plan_resp.action_output.split('\n')
    hints_removed_output = '\n'.join([line for line in output_lines if 'HINTS' not in line])
    plan_resp.action_output = hints_removed_output
    plan_str = str(plan_resp)

    # find the first write program
    prog_resps = [exp.response for exp in traj.exp_ls if exp.response.action_name == program_action_name]
    if len(prog_resps) > 1:
        raise ValueError('multiple write program resps detected, i assume only one there')
    prog_resp = deepcopy(prog_resps[0])

    # find the last revise code if there exists one
    revise_resps = [exp.response for exp in traj.exp_ls if exp.response.action_name == 'Revise code']
    if len(revise_resps) == 0:
        pass
    else:
        revise_resp = deepcopy(revise_resps[-1])
        # when substituting the output, i will simply do full replacement, assuming no text info should be preserved
        prog_resp.action_output = revise_resp.action_output

    expected_answer = f"=== Expected Answer form\n\n{sample['label']}\n\n" if include_expected_answer else ''

    example_prompt = (
        '=== Example question\n\n'
        f"{sample['input']}\n\n"
        f"{expected_answer}"
        '=== Example init thoughts and actions \n\n'
        f'{plan_str}\n\n'
        '===\n\n'
        f'{str(prog_resp)}'
    )

    return example_prompt


def make_icl_examples(
        samples,
        traj_key='traj',
        program_action_name='Write program',
        include_expected_answer=False,
        question_type_sampling=True,
):

    assert len(samples) > 0, 'no sample to make icl prompt from'

    # if samples are reclor type, then we return a dict of (qtype, prompts) and let prompt_prep_func to sample from
    if samples[0]['dataset'] == 'reclor' and question_type_sampling:
        qtype_prompt_dict = defaultdict(list)
        for sample in samples:
            qtype = sample['question_type']
            p = make_one_icl_prompt(
                sample,
                traj_key=traj_key,
                program_action_name=program_action_name,
                include_expected_answer=include_expected_answer
            )
            qtype_prompt_dict[qtype].append(p)
        return qtype_prompt_dict

    full_example_prompt = '\n\n'.join([
        make_one_icl_prompt(
            sample,
            traj_key=traj_key,
            program_action_name=program_action_name,
            include_expected_answer=include_expected_answer
        ) for sample in samples
    ])
    icl_prompt = (
        'below are example questions and the corresponding initial steps with thoughts and actions you can refer to\n\n'
        f'{full_example_prompt}'
        '\n\n==='
    )

    return icl_prompt


def traj2seq_simple(
        traj: Trajectory,
        no_exp_ok=False,
        remove_hints=True,
        add_resp_head=False,
        trainable_inds=None,
        add_boseos_to_icl_prompt=False,
        obs_trainable=True,
):
    """
        A simple way to make trainable seq from traj:

        For every exp in exp_ls:
            if exp is 'exec ok' or 'exec solved' then make resp and obs trainable
            if exp is 'exec wrong' then make obs trainable

        Additionally, for every resp and obs I will wrap it as <bos + resp/obs + eos>, so that the model learns when
        to stop

        Implication: 'exec ok' does not mean action is semantically right, it could be 'write program' that outputs
        trash code that runs without error. Training this with imperfect trajs is making model to imitate potentially
        non-optimal behaviors.

        To prevent this, you pass in trainable_inds, indicating inds of those trainable exps (the correct ones)

    :param traj:
    :param no_exp_ok: if set to True, I will ignore if exp_ls is None or [], this is used during inference to convert
        the inited traj to seqs that llama model could use to generate resp
    :param remove_hints: set to True to remove lines with substr "HINTS", should set to True for both training and
        inference
    :param add_resp_head: set to True to add append a resp str at the end (after the last obs), this is used in llama3
        inference, because it usually confuses when to generate obs and when to generate resp. Adding the resp head
        to hint llama3 to generate the resp
    :param trainable_inds: if provided, then each seq's trainabled flag is set to whatever is provided in this list
    :param add_boseos_to_icl_prompt: set to True to add llama boseos tokens to icl prompts, this is needed for base
        model llama3 icl inference to hopefully tell llama3 when to stop
    :param obs_trainable: set to True to make obs trainable, this is set to True for most training exps
    :return:
        [[s1, train_flag1], [s2, train_flag2], ...], where train_flag is bool
    """

    assert no_exp_ok or (all_exists(traj.exp_ls) and len(traj.exp_ls) > 0), 'give me valid traj'

    res = []

    init_prompt = traj.init_prompt
    if add_boseos_to_icl_prompt:
        init_prompt = init_prompt.replace(
            '=== observations ===',
            MyLlamaTokenizer.MY_EOS_TOKEN + '=== observations ==='
        )
        init_prompt = init_prompt.replace(
            '=== Example question',
            MyLlamaTokenizer.MY_EOS_TOKEN+'=== Example question'
        )
        tmp = """\n===

### Thought"""
        init_prompt = init_prompt.replace(
            tmp,
            MyLlamaTokenizer.MY_EOS_TOKEN + tmp
        )
        # NOTE don't switch with the above
        init_prompt = init_prompt.replace(
            '### Thought',
            MyLlamaTokenizer.MY_BOS_TOKEN + '### Thought'
        )
        init_prompt = init_prompt[:-3] + MyLlamaTokenizer.MY_EOS_TOKEN+'===\n'



    # add init prompt
    res.append([init_prompt, False])

    if all_exists(traj.exp_ls):
        for ind, exp in enumerate(traj.exp_ls):
            # add resp
            resp_trainable = (exp.exec_status == EXEC_OK) or (exp.exec_status == EXEC_SOLVED)
            # if you provide trainable_inds, then i will make those exp in the inds to be trainable
            if all_exists(trainable_inds):
                resp_trainable = ind in trainable_inds
            resp_str = ''.join([
                '=== response ===\n\n',
                MyLlamaTokenizer.MY_BOS_TOKEN,
                str(exp.response),
                '\n',
                MyLlamaTokenizer.MY_EOS_TOKEN,
                '\n'
            ])
            res.append([resp_str, resp_trainable])

            # add obs
            assert all_exists(exp.observation)
            obs_str = ''.join([
                '=== observations ===\n\n',
                MyLlamaTokenizer.MY_BOS_TOKEN,
                '\n\n'.join(str(fb) for fb in exp.observation),
                '\n',
                MyLlamaTokenizer.MY_EOS_TOKEN,
                '\n'
            ])
            res.append([obs_str, True if obs_trainable else False])

    # llama when trained with obs, tends to confuse itself about whether to generate obs or resp after an obs
    # here we do an ad-hoc fix, but appending the head of the resp, to hint llama that it should generate resp
    if add_resp_head:
        res.append([
            ''.join([
                '=== response ===\n\n',
                MyLlamaTokenizer.MY_BOS_TOKEN,
                '### Thought'
            ]),
            False
        ])

    # remove all lines that contains keyword "HINTS"
    if remove_hints:
        for e in res:
            res_lines = e[0].split('\n')
            hints_removed_res = '\n'.join([line for line in res_lines if 'HINTS' not in line])
            e[0] = hints_removed_res

    return res


def make_perfect_traj(sample, traj_key, runner):

    traj = sample[traj_key]
    if isinstance(traj, dict):
        traj = Trajectory.from_json(traj)

    datasetname = sample['dataset']
    dataset_traj_info = runner.datasetname2dataset_traj_info[datasetname]
    log_ls, trainable_inds = [], []

    # find the first plan, if exists one
    plan_ind, plan_exp = find_exp(traj, 'Plan', which_one='first')

    # find the first program
    prog_ind, prog_exp = find_exp(traj, dataset_traj_info.prog_action_name, which_one='first')
    if not all_exists(prog_exp):
        return None, None, [(None, None, None)]

    # find the last revise code after program
    # assuming exec_status is ok because every traj right now is solved
    code_ind, code_exp = find_exp(traj, 'Revise code', after=prog_ind, exec_status='exec ok',
                                  which_one='last')

    ans_after = code_ind if all_exists(code_ind) else prog_ind

    # find the last reason after code
    reason_ind, reason_exp = find_exp(traj, 'Reason', after=ans_after, which_one='last')

    # find the first answer after code
    ans_ind, ans_exp = find_exp(traj, 'Aggregate and answer', after=ans_after, which_one='first')
    assert all_exists(ans_exp)

    # find the last ans that is solved, we do this in case the first answer is wrong
    ans_ind2, ans_exp2 = find_exp(traj, 'Aggregate and answer', after=code_ind, exec_status='exec solved',
                                  which_one='last')
    assert all_exists(ans_exp2)

    # collect inds of trainable (meaning they are correct acts) exps
    trainable_inds = [ind for ind in [plan_ind, ans_after, reason_ind, ans_ind2] if all_exists(ind)]

    # make the perfect traj
    init_prompt = make_init_prompt(sample, runner.tactic_dict[dataset_traj_info.tactic_name])
    pj = Trajectory(init_prompt=init_prompt)

    # make plan
    pj.exp_ls = []
    if all_exists(plan_exp):
        plan_exp = deepcopy(plan_exp)
        pj.exp_ls.append(plan_exp)
        pj.exp_ls[-1].exec_status = 'exec ok'
        log_ls.append((plan_ind, plan_exp.action_type.action_name, plan_exp.exec_status))

    # make write program, replace output and feedback with those of revise code if possible
    prog_exp = deepcopy(prog_exp)
    if all_exists(code_exp):
        code_exp = deepcopy(code_exp)
        prog_exp.response.action_output = code_exp.response.action_output
        prog_exp.observation = code_exp.observation
        log_ls.append((
            f'{prog_ind}/{code_ind}',
            f'{prog_exp.action_type.action_name}/{code_exp.action_type.action_name}',
            f'{prog_exp.exec_status}/{code_exp.exec_status}'
        ))
    else:
        log_ls.append((
            prog_ind,
            prog_exp.action_type.action_name,
            prog_exp.exec_status
        ))
    pj.exp_ls.append(prog_exp)
    pj.exp_ls[-1].exec_status = 'exec ok'

    # make reason, if there exists one
    if all_exists(reason_exp):
        reason_exp = deepcopy(reason_exp)
        log_ls.append((
            reason_ind,
            reason_exp.action_type.action_name,
            reason_exp.exec_status
        ))
        pj.exp_ls.append(reason_exp)

    # make answer, replace output and feedback with those of the last one if possible
    ans_exp = deepcopy(ans_exp)
    if ans_exp.exec_status != 'exec solved':
        ans_exp2 = deepcopy(ans_exp2)
        ans_exp.response.action_output = ans_exp2.response.action_output
        ans_exp.observation = ans_exp2.observation
        log_ls.append((
            f'{ans_ind}/{ans_ind2}',
            f'{ans_exp.action_type.action_name}/{ans_exp2.action_type.action_name}',
            f'{ans_exp.exec_status}/{ans_exp2.exec_status}'
        ))
    else:
        log_ls.append((
            ans_ind,
            ans_exp.action_type.action_name,
            ans_exp.exec_status
        ))
    pj.exp_ls.append(ans_exp)
    pj.exp_ls[-1].exec_status = 'exec solved'

    return pj, trainable_inds, log_ls


def is_valid_answer(exp):
    """
        check if the exp has resp of type 'Aggregate and answer', if so, check if it is valid by checking the obs
        content, if valid, return True. This is used in inference where I make many LLMs' life easier because all of
        them have trouble following the right response format, so if I just stop the traj on every answer attempt, they
        will fail just because they did not answer it the right way. So if the answer is just invalid, I will not stop
        the traj and give them chances to correct it. Note this does not leak the answer.

        :return: True if valid, False if invalid, and None if no answer found
    """

    valid_ans = None
    if all_exists(exp) and (exp.response.action_name == 'Aggregate and answer'):
        # make sure the answer is properly parsed
        if all_exists(exp.observation) and len(exp.observation) > 0:
            # NOTE check if the answer is valid, this is ad-hoc, this should be returned by observers
            valid_ans = True
            for fb in exp.observation:
                if ('invalid answer' in fb.content) or ('invalid response format' in fb.content):
                    valid_ans = False
                    break
            # stop the traj, if the ans is valid, otherwise let it continue
            # we don't want to terminate just because it made formatting errors

    return valid_ans
