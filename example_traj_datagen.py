import numpy as np
import fire
from utils import all_exists
from tactic_and_execution.traj_datagen import batch_traj_datagen
from tactic_and_execution.traj_utils import manual_feedback
from utils.llm_requests import LLMRequestManager
from functools import partial


def prompt_prep_func(sample, prompt_dict, tgt_n, total_n):
    dn = sample['dataset']
    if dn == 'reclor':
        qtype_prompt_dict = prompt_dict[dn]
        sample_qtype = sample['question_type']
        if sample_qtype in qtype_prompt_dict:
            full_tgt_examples = qtype_prompt_dict[sample_qtype]
            if len(full_tgt_examples) < tgt_n:
                tgt_examples = full_tgt_examples
            else:
                tgt_examples = list(np.random.choice(full_tgt_examples, tgt_n, replace=False))
        else:
            tgt_examples = []
        actual_tgt_n = len(tgt_examples)
        other_examples = [example for k, ls in qtype_prompt_dict.items() for example in ls if k != sample_qtype]
        other_examples = list(np.random.choice(other_examples, total_n - actual_tgt_n, replace=False))
        full_example_prompt = '\n\n'.join(tgt_examples + other_examples)
        icl_prompt = (
            'below are example questions and the corresponding initial steps '
            'with thoughts and actions you can refer to\n\n'
            f'{full_example_prompt}'
            '\n\n==='
        )
        return icl_prompt
    else:
        return prompt_dict[dn]


def hint_feedback_func(sample, traj):
    if sample['dataset'] == 'proscript':
        exp = traj.exp_ls[-1]
        # make hints
        ind2name = sample['orig_data']['events']
        res = []
        res_simple = []
        for es in sample['label'].split('\n'):
            a, b = es.split('->')
            res.append(f"    # HINTS: '{a}'->'{b}', first {ind2name[a]}; then {ind2name[b]}")
            res_simple.append(f"# HINTS: '{a}'->'{b}'")
        hint_str = '\n' + '\n'.join(res[:len(res)])
        hint_str_simple = '\n'.join(res_simple)

        # inject hints
        if all_exists(exp.action_type):
            an, status = exp.action_type.action_name, exp.exec_status
            if (len(traj.exp_ls) == 1) and (an == 'Plan'):
                traj.exp_ls[-1].response.action_output += hint_str
                traj.exp_ls[-1].response.raw += hint_str
            elif (status == 'exec wrong') and (
                    (an == 'Build graph model') or
                    (an == 'Revise code') or
                    (an == 'Aggregate and answer')
            ):
                manual_feedback(traj, content=hint_str_simple, observer='user', feedback_status='feedback wrong')
    elif sample['dataset'] == 'folio':
        exp = traj.exp_ls[-1]
        hint_str = '\n'.join([f'HINTS {s}' for s in sample['orig_data']['premise_fol_ls']])
        # inject hints
        if all_exists(exp.action_type):
            an, status = exp.action_type.action_name, exp.exec_status
            if (status == 'exec wrong') and (
                    (an == 'Build FOL model') or
                    (an == 'Revise code')
            ):
                manual_feedback(traj, content=hint_str, observer='user', feedback_status='feedback wrong')

    elif sample['dataset'] == 'reclor':
        pass
        # You can uncomment below to give hint to reclor examples, though this typically leads to trivial program so
        # I don't recommend it
        # exp = traj.exp_ls[-1]
        # hint_str = f'\n # HINTS answer {sample["label"]}'
        # if all_exists(exp.action_type):
        #     an, status = exp.action_type.action_name, exp.exec_status
        #     if (status == 'exec wrong') and (
        #             (an == 'Write program') or
        #             (an == 'Revise code')
        #     ):
        #         manual_feedback(traj, content=hint_str, observer='user', feedback_status='feedback wrong')
        #
        #     if an == 'Plan':
        #         manual_feedback(traj, content=hint_str, observer='user', feedback_status='feedback ok')


def example_traj_datagen(api_key_file_path: str):
    """
        this example showcases the trajectory generation using openai batch api on a tiny dataset of 4 samples
        each from one dataset (excluding those that provides icl examples)

        Once this is done, checkout notebook/inspect-traj.ipynb to see how to inspect the generated trajs

    :param api_key_file_path: the path to the api key file with every line being <type>: <key> where <type> should be
        one of the following: openai, gemini, mistral, anthropic, cohere
    """
    total_n = 5
    tgt_n = 2
    gpt4, gpt4o = LLMRequestManager.model_gpt4_turbo_0409, LLMRequestManager.model_gpt4o

    prompt_prep_func_with_n = partial(prompt_prep_func, tgt_n=tgt_n, total_n=total_n)

    def filter_func(e):
        # filter out those samples with icl trajs
        return 'traj' not in e

    batch_traj_datagen(
        'data/traj_toydata.json',
        save_path='data/traj_toydata-output.json',
        input_batch_path='data/tmp_in_batch.json',
        output_batch_path='data/tmp_out_batch.json',
        inprogress_key='traj_in_progress',
        save_key='icl_gpt4_traj',
        # for difficult ones, we use gpt4 for the first two rounds, and gpt4o for the rest
        model_name={
            'gsm8k': [gpt4, gpt4, gpt4],
            'proscript': [gpt4, gpt4, gpt4o],
            'reclor': [gpt4, gpt4, gpt4, gpt4o],
            'folio': [gpt4, gpt4, gpt4o]
        },
        icl_traj_key='traj',
        icl_prompt_dict=None,
        icl_prompt_prep_func=prompt_prep_func_with_n,
        rerun_on_failed=False,
        max_rerun=0,
        rerun_retain_until={
            'gsm8k': 0,
            'proscript': 0,
            'reclor': 0,
            'folio': 0
        },
        max_n_requests=7,
        use_icl_before_round=2,
        filter_func=filter_func,
        manual_feedback_func=hint_feedback_func,
        tactics_dir='data/tactics',
        api_key_path=api_key_file_path,
    )


if __name__ == '__main__':
    fire.Fire(example_traj_datagen)