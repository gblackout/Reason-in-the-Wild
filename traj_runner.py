from tactic_and_execution.thought_action_observation_data_structure import *
from tactic_and_execution.execution import Execution
from typing import Optional, List, Dict, Union, Callable
import os
from os.path import join as joinpath
from dataclasses import dataclass
from tactic_and_execution.observer import BaseObserver
from tactic_and_execution.any_program_observer import AnyProgramObserver
from tactic_and_execution.fol_z3_observer import FOLZ3Observer
from tactic_and_execution.math_observer import MathObserver
from tactic_and_execution.graph_observer import GraphObserver
from tactic_and_execution.routing_observer import RoutingObserver
from functools import partial
from copy import deepcopy
from tactic_and_execution.traj_utils import manual_feedback, find_exp, is_valid_answer, make_init_prompt
import time
from utils import make_parent_dirs


@dataclass
class DatasetTrajInfo:
    tactic_name: Optional[str] = None
    prog_action_name: Optional[str] = None
    observer_class: Optional[BaseObserver] = None
    obs_feedback_kwargs: Optional[Dict] = None


class TrajRunner:

    datasetname2dataset_traj_info = {
        'reclor': DatasetTrajInfo(
            tactic_name='any_program',
            prog_action_name='Write program',
            observer_class=AnyProgramObserver,
            obs_feedback_kwargs={'gt_answer': 'SAMPLE_LABEL', 'answer_set': ['1', '2', '3', '4']}
        ),
        'gsm8k': DatasetTrajInfo(
            tactic_name='math',
            prog_action_name='Build math model',
            observer_class=MathObserver,
            obs_feedback_kwargs={'gt_answer': [partial, MathObserver.check_answer, 'SAMPLE_LABEL']}
        ),
        'proscript': DatasetTrajInfo(
            tactic_name='graph',
            prog_action_name='Build graph model',
            observer_class=GraphObserver,
            # we consider proscript answer correct as long as the f score is above 0.6, this is because this dataset
            # is inherently noisy and ambiguous
            obs_feedback_kwargs={'gt_answer': [partial, GraphObserver.check_answer, 'SAMPLE_LABEL', 0.6]}
        ),
        'folio': DatasetTrajInfo(
            tactic_name='predicate_logic_z3',
            prog_action_name='Build FOL model',
            observer_class=FOLZ3Observer,
            obs_feedback_kwargs={'gt_answer': 'SAMPLE_LABEL', 'answer_set': ['Agree', 'Contradict', 'Uncertain']}
        ),
        'hyb': DatasetTrajInfo(
            tactic_name='main_routing',
            prog_action_name=None,
            observer_class=RoutingObserver,
            obs_feedback_kwargs={'gt_answer': 'SAMPLE_LABEL', 'answer_set': 'SAMPLE_ANSWER_SET'}
        )
    }

    def __init__(self, tactics_dir: str = 'data/tactics'):

        # load tactics
        self.tactic_dict = {}
        for fn in os.listdir(tactics_dir):
            fp = joinpath(tactics_dir, fn)
            if os.path.isdir(fp):
                fpp = joinpath(fp, 'tactic.txt')
                if os.path.isfile(fpp):
                    with open(fpp, 'r') as f:
                        tactic = f.read()
                        self.tactic_dict[fn] = tactic

        # init obs
        self.obs_dict = {}
        for datasetname, dataset_traj_info in self.datasetname2dataset_traj_info.items():
            obs = dataset_traj_info.observer_class(self.tactic_dict[dataset_traj_info.tactic_name])
            self.obs_dict[datasetname] = obs

        self.exec = Execution()

    def init_one_traj(
            self,
            sample: Dict,
            icl_prompt: str = '',
            use_icl_before_round: int = -1,
            sample_input_key: Optional[str] = None,
            baby_prompt: str = '',
            answer_check: bool = True,
            **metadata_kwargs,
    ):
        datasetname = sample['dataset']
        dataset_traj_info = self.datasetname2dataset_traj_info[datasetname]
        obs = self.obs_dict[datasetname]
        tactic = self.tactic_dict[dataset_traj_info.tactic_name]

        init_prompt = make_init_prompt(
            sample,
            tactic,
            sample_input_key=sample_input_key
        )

        traj = Trajectory(init_prompt=init_prompt + icl_prompt)
        traj.metadata = {
            'use_icl_before_round': use_icl_before_round,
            'init_prompt': init_prompt,
            'init_icl_prompt': init_prompt + icl_prompt,
            'init_baby_prompt': init_prompt + baby_prompt,
            **metadata_kwargs
        }

        # load feedback func args, and replace placeholders labels with sample labels
        fb_kwargs = deepcopy(dataset_traj_info.obs_feedback_kwargs)
        if fb_kwargs['gt_answer'] == 'SAMPLE_LABEL':
            fb_kwargs['gt_answer'] = sample['label']
        elif isinstance(fb_kwargs['gt_answer'], list):
            for arg_ind, arg in enumerate(fb_kwargs['gt_answer']):
                if arg == 'SAMPLE_LABEL':
                    fb_kwargs['gt_answer'][arg_ind] = sample['label']
            fb_kwargs['gt_answer'] = fb_kwargs['gt_answer'][0](*fb_kwargs['gt_answer'][1:])
        else:
            raise ValueError(f'i don\'t know how to handle this fb_kwargs {fb_kwargs["gt_answer"]}')
        # replace answer_set placeholders for hyb data
        if 'answer_set' in fb_kwargs:
            if fb_kwargs['answer_set'] == 'SAMPLE_ANSWER_SET':
                fb_kwargs['answer_set'] = [f'{ind+1}' for ind in range(len(sample['options']))]

        # specify if answer is checked. Usually you don't want to check it if its a subtraj
        fb_kwargs['answer_check'] = answer_check

        extra_feedback_funcs = [partial(obs.give_feedback, **fb_kwargs)]

        return traj, obs, extra_feedback_funcs

    def step(
            self,
            request_func: Callable,
            traj: Trajectory,
            max_n_requests: int = 7,
            max_n_consecutive_wrong_actions: int = 3,
            max_n_wrong_actions: int = 10,
            termination_func: Optional[Callable] = None,
            exec_status_func: Optional[Callable] = None,
            extra_feedback_funcs: Optional[List[Callable]] = None,
            give_traj_to_request_func: bool = False,
            fuzzy_parse: bool = False
    ):
        # this is used when request_func is local model generation func, which requires to access traj to put boseos
        # in it
        if give_traj_to_request_func:
            request_func = partial(request_func, traj=traj)

        new_messages, exp, traj_status = self.exec.step(
            request_func=request_func,
            trajectory=traj,
            max_n_requests=max_n_requests,
            max_n_consecutive_wrong_actions=max_n_consecutive_wrong_actions,
            max_n_wrong_actions=max_n_wrong_actions,
            termination_func=termination_func,
            exec_status_func=exec_status_func,
            extra_feedback_funcs=extra_feedback_funcs,
            fuzzy_parse=fuzzy_parse
        )

        return new_messages, exp, traj_status

    def run_one_traj(
            self,
            request_func: Callable,
            traj: Trajectory,
            max_n_requests: int = 7,
            max_n_consecutive_wrong_actions: int = 3,
            max_n_wrong_actions: int = 10,
            termination_func: Optional[Callable] = None,
            exec_status_func: Optional[Callable] = None,
            extra_feedback_funcs: Optional[List[Callable]] = None,
            verbose: bool = True,
            give_traj_to_request_func: bool = False,
            fuzzy_parse: bool = False,
            stop_on_answer: bool = False,
            use_baby_prompt_after_icl: bool = False,
            req_wait: int = -1
    ):
        traj_status = 'S'
        round = 0
        while (traj_status != TRAJ_TERMINATED) and (traj_status != TRAJ_REQUEST_FAILED):
            new_messages, exp, traj_status = self.step(
                request_func=request_func,
                traj=traj,
                max_n_requests=max_n_requests,
                max_n_consecutive_wrong_actions=max_n_consecutive_wrong_actions,
                max_n_wrong_actions=max_n_wrong_actions,
                termination_func=termination_func,
                exec_status_func=exec_status_func,
                extra_feedback_funcs=extra_feedback_funcs,
                give_traj_to_request_func=give_traj_to_request_func,
                fuzzy_parse=fuzzy_parse
            )
            if req_wait > 0:
                time.sleep(req_wait)

            round += 1
            if round == traj.metadata['use_icl_before_round']:
                if use_baby_prompt_after_icl:
                    traj.init_prompt = traj.metadata['init_baby_prompt']
                else:
                    traj.init_prompt = traj.metadata['init_prompt']

            if verbose:
                if all_exists(exp):
                    print(exp.nice_str())

            # stop the traj if the ans is valid, otherwise let it continue
            # we don't want to terminate just because it made formatting errors
            if stop_on_answer:
                if is_valid_answer(exp) is True:
                    break

        return traj, traj_status

    def run_one_routing_traj(
            self,
            request_func: Callable,
            sample: Dict, # different from single-form traj, i need the sample to access the orig questions
            traj: Trajectory,
            icl_prompt_dict: Dict,
            sub_traj_init_kwargs: Dict,
            sub_traj_run_kwargs: Dict,
            max_n_requests: int = 7,
            max_n_consecutive_wrong_actions: int = 3,
            max_n_wrong_actions: int = 10,
            termination_func: Optional[Callable] = None,
            exec_status_func: Optional[Callable] = None,
            extra_feedback_funcs: Optional[List[Callable]] = None,
            verbose: bool = True,
            give_traj_to_request_func: bool = False,
            fuzzy_parse: bool = False,
            stop_on_answer: bool = False,
            use_baby_prompt_after_icl_in_subtraj: bool = False,
            req_wait: int = -1,
            task_adapter_name: Optional[str] = None,
    ):
        sub_traj_status = 'S'
        traj_status = 'S'
        round = 0
        while (
                (traj_status != TRAJ_TERMINATED) and
                (traj_status != TRAJ_REQUEST_FAILED) and
                # if any of the subtraj has request failed then we stop the whole traj
                (sub_traj_status != TRAJ_REQUEST_FAILED)
        ):

            # the case where the model is local llama and has a task adapter to be switched for the subproblem
            if all_exists(task_adapter_name):
                routing_request_func = partial(request_func, which_adapter=None, is_hyb=True)
            else:
                routing_request_func = request_func

            new_messages, exp, traj_status = self.step(
                request_func=routing_request_func,
                traj=traj,
                max_n_requests=max_n_requests,
                max_n_consecutive_wrong_actions=max_n_consecutive_wrong_actions,
                max_n_wrong_actions=max_n_wrong_actions,
                termination_func=termination_func,
                exec_status_func=exec_status_func,
                extra_feedback_funcs=extra_feedback_funcs,
                give_traj_to_request_func=give_traj_to_request_func,
                fuzzy_parse=fuzzy_parse
            )
            if req_wait > 0:
                time.sleep(req_wait)

            # NOTE we always use baby prompt for routing traj, the flag use_baby_prompt_after_icl_in_subtraj indicates
            # if this is used in subtraj or not
            round += 1
            if round == traj.metadata['use_icl_before_round']:
                traj.init_prompt = traj.metadata['init_prompt']

            if verbose:
                if all_exists(exp):
                    print(exp.nice_str())

            # stop the traj if the ans is valid, otherwise let it continue
            # we don't want to terminate just because it made formatting errors
            if stop_on_answer:
                if is_valid_answer(exp) is True:
                    break

            if (exp is None) or (exp.action_type is None):
                continue

            # ================== run sub traj ==================
            is_calltactic = 'Call tactic' in exp.action_type.action_name
            should_terminate = self.exec.check_should_terminate(
                trajectory=traj,
                max_n_requests=max_n_requests,
                max_n_consecutive_wrong_actions=max_n_consecutive_wrong_actions,
                max_n_wrong_actions=max_n_wrong_actions,
                termination_func=termination_func,
            )

            if should_terminate or (not is_calltactic):
                continue

            if (
                    (traj.metadata is None) or
                    ('subtrajs' not in traj.metadata) or
                    (len(traj.metadata['subtrajs']) == 0) or
                    # the case somehow this subtraj_info already has a run traj
                    ('subtraj' in traj.metadata['subtrajs'][-1])
            ):
                manual_feedback(
                    traj,
                    'No valid subproblem found',
                    observer=TACTIC_RUNNER, feedback_status=FEEDBACK_WRONG
                )
                continue

            subtraj_info = traj.metadata['subtrajs'][-1]
            oind = subtraj_info['oind']
            oqind = sample['option_ind2qls'][oind]
            oq = sample['qls'][oqind]
            is_correct_option = str(oind + 1) == sample['label']
            label = oq['label']
            # the correct label of reclor needs to be modified
            if oq['dataset'] == 'reclor':
                label = '1' if is_correct_option else '2'
            # this is the datasetname inferred from the tactic_name which is in turn inferred from the model
            # it could potentially mismatch with that of the gt dn. This is used in init_one_traj to load the
            # corresponding tactic regardless if it is a mismatch because the model does not know
            # NOTE this is assuming one-one mapping between tactic and dn
            chosen_datasetname = [
                dn for dn, info in self.datasetname2dataset_traj_info.items() if
                info.tactic_name == subtraj_info['tactic_name']
            ][0]

            subtraj_sample = {
                'input': subtraj_info['subproblem'],
                'dataset': chosen_datasetname,
                'label': label # this is the gt label of the oq, but it is not checked against ans in subtraj
            }

            k = f"{chosen_datasetname}_baby"
            baby_prompt = icl_prompt_dict[k] if k in icl_prompt_dict else ''
            sub_traj, sub_obs, sub_extra_feedback_funcs = self.init_one_traj(
                sample=subtraj_sample,
                icl_prompt=icl_prompt_dict[chosen_datasetname],
                use_icl_before_round=sub_traj_init_kwargs['use_icl_before_round'],
                baby_prompt=baby_prompt,
                answer_check=False # for subtraj, we dont check answers
            )

            # the case where the model is local llama and has a task adapter to be switched for the subproblem
            if all_exists(task_adapter_name):
                subproblem_request_func = partial(request_func, which_adapter=task_adapter_name, is_hyb=True)
            else:
                subproblem_request_func = request_func

            sub_traj, sub_traj_status = self.run_one_traj(
                request_func=subproblem_request_func,
                traj=sub_traj,
                extra_feedback_funcs=sub_extra_feedback_funcs,
                use_baby_prompt_after_icl=use_baby_prompt_after_icl_in_subtraj,
                stop_on_answer=True,
                req_wait=req_wait,
                **sub_traj_run_kwargs
            )

            # save subtraj
            subtraj_info['subtraj'] = sub_traj.to_dict()

            # put output back to the main traj
            content_out = None
            if all_exists(sub_traj.exp_ls) and (len(sub_traj.exp_ls) > 0):
                _, ans_exp = find_exp(sub_traj, 'Aggregate and answer', which_one='last')
                if all_exists(ans_exp) and all_exists(ans_exp.response) and all_exists(ans_exp.response.action_output):
                    content_out = ans_exp.response.action_output
            if content_out is None:
                content = 'Tactic execution failed. Tactic output:\nruntime error, please try again.'
            else:
                content = f'Tactic execution successful. Tactic output:\n{content_out}'

            manual_feedback(traj, content=content, observer=TACTIC_RUNNER, feedback_status='feedback ok')

        return traj, traj_status, sub_traj_status

    def run_all_trajs(
            self,
            data: Union[str, List[Dict]],
            save_key: str,
            filter_func: Optional[Callable] = None,
            request_func: Optional[Callable] = None,
            use_icl_before_round: int = -1,
            # traj termination params
            max_n_requests: int = 7,
            max_n_consecutive_wrong_actions: int = 3,
            max_n_wrong_actions: int = 10,
            termination_func: Optional[Callable] = None,
            exec_status_func: Optional[Callable] = None,
            verbose: bool = True,
            give_traj_to_request_func: bool = False,
            save_every: int = 3,
            save_path: Optional[str] = None,
            tqdm_func: Optional[Callable] = None,
            fuzzy_parse: bool = False,
            stop_on_answer: bool = False,
            use_baby_prompt_after_icl: bool = False,
            # hyb_run_traj_kwargs
            hyb_run: bool = False,
            icl_prompt_dict: Optional[Dict] = None,
            sub_traj_init_kwargs: Optional[Dict] = None,
            sub_traj_run_kwargs: Optional[Dict] = None,
            req_wait: int = -1,
            task_adapter_name: Optional[str] = None
    ):
        # load data
        if isinstance(data, str):
            with open(data, 'r') as f:
                data = json.load(f)

        make_parent_dirs(save_path)

        # prep funcs
        _filter_func = filter_func if all_exists(filter_func) else lambda _: True

        pbar = tqdm_func(data, leave=False) if all_exists(tqdm_func) else None
        update_bar = pbar.update if all_exists(tqdm_func) else lambda: None
        cnt = 0
        for ind, sample in enumerate(data):
            if not _filter_func(sample):
                update_bar()
                continue

            icl_prompt = icl_prompt_dict[sample['dataset']]
            k = f"{sample['dataset']}_baby"
            baby_prompt = icl_prompt_dict[k] if k in icl_prompt_dict else ''
            traj, obs, extra_feedback_funcs = self.init_one_traj(
                sample=sample,
                icl_prompt=icl_prompt,
                use_icl_before_round=use_icl_before_round,
                baby_prompt=baby_prompt
            )

            if hyb_run:
                traj, traj_status, sub_traj_status = self.run_one_routing_traj(
                    request_func=request_func,
                    sample=sample,
                    traj=traj,
                    icl_prompt_dict=icl_prompt_dict,
                    sub_traj_init_kwargs=sub_traj_init_kwargs,
                    sub_traj_run_kwargs=sub_traj_run_kwargs,
                    max_n_requests=max_n_requests,
                    max_n_consecutive_wrong_actions=max_n_consecutive_wrong_actions,
                    max_n_wrong_actions=max_n_wrong_actions,
                    termination_func=termination_func,
                    exec_status_func=exec_status_func,
                    extra_feedback_funcs=extra_feedback_funcs,
                    verbose=verbose,
                    give_traj_to_request_func=give_traj_to_request_func,
                    fuzzy_parse=fuzzy_parse,
                    stop_on_answer=stop_on_answer,
                    use_baby_prompt_after_icl_in_subtraj=use_baby_prompt_after_icl,
                    req_wait=req_wait,
                    task_adapter_name=task_adapter_name
                )
                should_save = (traj_status != TRAJ_REQUEST_FAILED) and (sub_traj_status != TRAJ_REQUEST_FAILED)
            else:
                traj, traj_status = self.run_one_traj(
                    request_func=request_func,
                    traj=traj,
                    max_n_requests=max_n_requests,
                    max_n_consecutive_wrong_actions=max_n_consecutive_wrong_actions,
                    max_n_wrong_actions=max_n_wrong_actions,
                    termination_func=termination_func,
                    exec_status_func=exec_status_func,
                    extra_feedback_funcs=extra_feedback_funcs,
                    verbose=verbose,
                    give_traj_to_request_func=give_traj_to_request_func,
                    fuzzy_parse=fuzzy_parse,
                    stop_on_answer=stop_on_answer,
                    use_baby_prompt_after_icl=use_baby_prompt_after_icl,
                    req_wait=req_wait
                )
                should_save = traj_status != TRAJ_REQUEST_FAILED

            sample[save_key] = traj.to_dict() if should_save else None

            cnt += 1
            if (cnt % save_every == 0) and all_exists(save_path):
                with open(save_path, 'w') as f:
                    json.dump(data, f)

            update_bar()

        with open(save_path, 'w') as f:
            json.dump(data, f)
