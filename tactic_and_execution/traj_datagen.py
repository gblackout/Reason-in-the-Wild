from utils.llm_requests import LLMRequestManager, BatchReqManager
from traj_runner import TrajRunner
from tactic_and_execution.thought_action_observation_data_structure import *
from copy import deepcopy
from typing import Callable
from tactic_and_execution.traj_utils import make_icl_examples


def default_traj_update_func(
        sample: Dict,
        runner: TrajRunner,
        inprogress_key: str,
        save_key: str,
        rerun_on_failed: bool = False,
        max_rerun: int = 5,
        rerun_retain_until: int = 0,
        **termination_kwargs
):
    if inprogress_key not in sample:
        return False

    traj = Trajectory.from_json(sample[inprogress_key])
    should_terminate = runner.exec.check_should_terminate(
        trajectory=traj,
        **termination_kwargs
    )

    def finish_run():
        sample[save_key] = traj.to_dict()
        del sample[inprogress_key]
        return False

    # if traj unfinish
    if not should_terminate:
        return True

    # if traj finished and solved
    if traj.cur_status() == EXEC_SOLVED:
        return finish_run()

    # if traj finished and failed and not rerun_on_failed
    if not rerun_on_failed:
        return finish_run()

    # if traj finished and failed and rerun_on_failed
    assert isinstance(traj.metadata, dict), 'metadata missing, you need to properly init traj using traj runner'

    if 'failed_exp_ls_ls' in traj.metadata:
        tgt_ls = traj.metadata['failed_exp_ls_ls']
    else:
        tgt_ls = []
        traj.metadata['failed_exp_ls_ls'] = tgt_ls
    cur_rerun_cnt = len(tgt_ls)
    if cur_rerun_cnt >= max_rerun:
        return finish_run()

    # now we want to rerun, save the old exp_ls and truncate it
    exp_ls = deepcopy(traj.exp_ls)
    tgt_ls.append([e.to_dict() for e in exp_ls])
    traj.exp_ls = traj.exp_ls[:rerun_retain_until]
    if len(traj.exp_ls) == 0:
        traj.exp_ls = None
    # update the traj obj in sample
    sample[inprogress_key] = traj.to_dict()

    return True


def batch_traj_datagen(
        data: Union[str, Dict],
        save_path: str,
        input_batch_path: str,
        output_batch_path: str,
        inprogress_key: str,
        save_key: str,
        model_name: Union[str, Dict],
        icl_traj_key: Optional[str] = None,
        icl_prompt_dict: Optional[Dict] = None,
        icl_prompt_prep_func: Optional[Callable] = None,
        rerun_on_failed: bool = False,
        max_rerun: int = 0,
        rerun_retain_until: Union[int, Dict] = 0,
        max_n_requests: int = 7,
        max_n_consecutive_wrong_actions: int = 3,
        max_n_wrong_actions: int = 10,
        use_icl_before_round: int = 2,
        filter_func: Optional[Callable] = None,
        manual_feedback_func: Optional[Callable] = None,
        tactics_dir: str = 'data/tactics',
        api_key_path: str = 'llm-api-keys.txt',
):
    """
        this is the main function I use to generate trajectories using OpenAI batch API, you give the path <data> to
        a set of samples of the form {'dataset': dn, 'input': input, ...}, and I will generate the problem-solving
        trajectories for it.

    :param data: path to the set of samples
    :param save_path: path to saving the samples
    :param input_batch_path: path to saving the tmp input batch files, note it might create multiple files with this as
        prefix, because a single file cannot have different models, I need to split it for every models
    :param output_batch_path: path to saving the tmp output batch files downloaded from openai, same as input_batch_path
        it might save mutliple files
    :param inprogress_key: i will save the in progress traj to sample[inprogress_key]
    :param save_key: i will save the final traj to sample[save_key], this needs to be different from inprogress_key
    :param model_name: the openai model to be used for generation, could be a single str or a dict of list of the form
        {dn1: [model_name1, model_name2, ...], ...}
    :param icl_traj_key: generation needs icl examples, you can put example trajs in sample[icl_traj_key], which will
        be gathered per dataset and compiled to icl prompt
    :param icl_prompt_dict: alternatively, you can give icl_prompt_dict directly of the form {dn1: prompt1, dn2: ...}
    :param icl_prompt_prep_func: a func called during traj init, accepting sample and icl_prompt_dict, typically for
        well-defined datasets this is simply icl_prompt_dict[dn] but for reclor it is needed, note the prompt structure
        for reclor is different because we need to sample examples by qtype, so the examples saved to _icl_prompt_dict
        using make_icl_examples are dict of lists rather than str, and the sampling happens in the icl_prompt_prep_func
    :param rerun_on_failed: if a traj failed, do you want to rerun it?
    :param max_rerun: if rerun_on_failed then how many tries do you want to give?
    :param rerun_retain_until: if rerun_on_failed, do you want to retain part of the trajs? if you put 2, that means
        the first two steps are kept and the rest are discarded in rerun. I usually use it for gsm8k where I do first
        2 steps with gpt4 and the rest with gpt4o to save money, but gpt4o makes more errors, that's why I could reuse
        the part generated by gpt4
    :param max_n_requests: the max length of a traj
    :param max_n_consecutive_wrong_actions: the max consecutive_wrong_actions allowed in a traj
    :param max_n_wrong_actions: the max total wrong_actions allowed in a traj
    :param use_icl_before_round: if set to k, then for the icl prompt is used only for the first k round
    :param filter_func: a func called before init that accepts the sample, return True if this sample needs to be
        generated
    :param manual_feedback_func: a func called after every step of traj that accepts sample and traj, this gives you
        chance to append feedbacks to the traj observations, and during generation, this is typically hints
    :param tactics_dir: path to the tactic folder used to init TrajRunner
    :param api_key_path: paht to the api key file used to init LLMRequestManager
    :return:
    """
    runner = TrajRunner(tactics_dir=tactics_dir)
    manager = LLMRequestManager(api_key_path)
    batch_manager = BatchReqManager(manager)

    # load data
    if isinstance(data, str):
        with open(data, 'r') as f:
            data = json.load(f)

    _filter_func = filter_func if all_exists(filter_func) else lambda _: True
    _manual_feedback_func = manual_feedback_func if all_exists(manual_feedback_func) else lambda _, __: None
    _icl_prompt_dict = dict((dn, '') for dn in runner.datasetname2dataset_traj_info)
    _prompt_prep_func = lambda s, prompt_dict: prompt_dict[s['dataset']]
    # in case model_name is str, then we use the same model for all requests
    # in case model_name is dict, then I expect it is a dict of the form {dn: [mn1, mn2, ...], ...}
    request_schedule_dict = (
        dict((dn, [model_name]) for dn in runner.datasetname2dataset_traj_info)
        if isinstance(model_name, str)
        else model_name
    )

    assert not all_exists(icl_traj_key, icl_prompt_dict), \
        'If you use icl_traj_key then you should not give icl_prompt_dict'
    if all_exists(icl_prompt_dict):
        _icl_prompt_dict = icl_prompt_dict
    if all_exists(icl_prompt_prep_func):
        _prompt_prep_func = icl_prompt_prep_func
    if all_exists(icl_traj_key):
        manual_examples = [e for e in data if icl_traj_key in e]
        for dn in runner.datasetname2dataset_traj_info:
            dataset_manual_examples = [e for e in manual_examples if e['dataset'] == dn]
            # we make icl prompt for those having manual_examples otherwise just leave it as ''
            if len(dataset_manual_examples) > 0:
                _icl_prompt_dict[dn] = make_icl_examples(
                    dataset_manual_examples,
                    program_action_name=runner.datasetname2dataset_traj_info[dn].prog_action_name,
                    include_expected_answer=True
                )

    # init traj for all data
    full_progress_batch = []
    for ind, sample in enumerate(data):
        if not _filter_func(sample):
            continue

        traj, obs, extra_feedback_funcs = runner.init_one_traj(
            sample=sample,
            icl_prompt=_prompt_prep_func(sample, _icl_prompt_dict),
            use_icl_before_round=use_icl_before_round,
            request_schedule=request_schedule_dict[sample['dataset']]
        )

        sample[inprogress_key] = traj.to_dict()
        full_progress_batch.append([sample, obs, extra_feedback_funcs])

    active_batch = full_progress_batch
    print(f'{len(active_batch)}/{len(full_progress_batch)} in progress')
    while len(active_batch) > 0:

        cnt = 0
        with open(input_batch_path, 'w') as f:
            for ind, (sample, _, _) in enumerate(active_batch):

                traj = Trajectory.from_json(sample[inprogress_key])
                cur_round = 0 if traj.exp_ls is None else len(traj.exp_ls)
                use_icl_before_round = traj.metadata['use_icl_before_round']
                request_schedule = traj.metadata['request_schedule']
                request_model_name = request_schedule[min(cur_round, len(request_schedule) - 1)]

                # NOTE this init_prompt replace will only affect the batch jsonl file, since we do not save traj back
                # NOTE to sample via sample[inprogress_key] = traj.to_dict(), so the traj.init_prompt in the final
                # NOTE save_path will still be whatever is inited above using runner.init_one_traj
                if cur_round >= use_icl_before_round:
                    traj.init_prompt = traj.metadata['init_prompt']
                else:
                    traj.init_prompt = traj.metadata['init_icl_prompt']

                entry = batch_manager.make_jsonl_entry(f'traj-{cnt}', request_model_name, traj.to_messages())

                cnt += 1
                s = json.dumps(entry)
                f.write(s + '\n')

        tracker = batch_manager.upload_and_start_batch(input_batch_path)
        status = batch_manager.watch_and_download_batches(tracker, output_batch_path)
        if status != 'completed':
            print(status)
            break
        batch_resp = batch_manager.batch2messages(tracker=tracker)

        for ind, (sample, _, extra_feedback_funcs) in enumerate(active_batch):

            traj = Trajectory.from_json(sample[inprogress_key])
            new_messages = traj.to_messages()
            new_messages.append(batch_resp[ind])

            _, exp, traj_status = runner.exec.req_step(
                trajectory=traj,
                new_messages=new_messages,
                extra_feedback_funcs=extra_feedback_funcs,
            )

            # apply manual feedback if necessary
            _manual_feedback_func(sample, traj)

            sample[inprogress_key] = traj.to_dict()

        active_batch = []
        for ind, (sample, obs, extra_feedback_funcs) in enumerate(full_progress_batch):

            rerun_retain_until_int = rerun_retain_until if isinstance(rerun_retain_until, int) \
                else rerun_retain_until[sample['dataset']]
            is_active = default_traj_update_func(
                sample=sample,
                runner=runner,
                inprogress_key=inprogress_key,
                save_key=save_key,
                rerun_on_failed=rerun_on_failed,
                max_rerun=max_rerun,
                rerun_retain_until=rerun_retain_until_int,
                max_n_requests=max_n_requests,
                max_n_consecutive_wrong_actions=max_n_consecutive_wrong_actions,
                max_n_wrong_actions=max_n_wrong_actions
            )

            if is_active:
                active_batch.append([sample, obs, extra_feedback_funcs])

        print(f'{len(active_batch)}/{len(full_progress_batch)} in progress')

        with open(save_path, 'w') as f:
            json.dump(data, f)
