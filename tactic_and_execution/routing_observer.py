import re

from .observer import BaseObserver
from .thought_action_observation_data_structure import Trajectory, Feedback
from .consts import *
from utils import all_exists, split_string_with_separators
from typing import Callable

ROUTING_RESP_TEMPLATE = """### option
<index of the option>
### subproblem
<subproblem>
"""


class RoutingObserver(BaseObserver):

    fuzz_kw_ls = [
        ['### option', '## option', '# option'],
        ['### subproblem', '## subproblem', '# subproblem']
    ]

    callname2tacticname = {
        'Call tactic: directed graph': 'graph',
        'Call tactic: math': 'math',
        'Call tactic: formal logic z3': 'predicate_logic_z3',
        'Call tactic: general program': 'any_program'
    }

    @staticmethod
    def check_answer(gt, pred, *args, **kwargs):
        raise NotImplementedError

    def give_feedback(
        self,
        trajectory: Trajectory,
        gt_answer=None,
        answer_set=None,
        which_exp=None,
        answer_check=True,
        **kwargs
    ):
        feedbacks = []
        if all_exists(which_exp):
            assert isinstance(which_exp, int), 'give me the int index of the exp you want feedback for'
            exp_ind = which_exp
        else:
            exp_ind = -1
        exp = trajectory.exp_ls[exp_ind]

        # ========================== parsing action format ==========================
        parsed_action_type = self.parse_action_type(exp)
        # if parsing failed -> invalid action name
        if parsed_action_type is None:
            action_names = '\n'.join([k for k in self.action_space.action_dict])
            fb = Feedback(
                observer=ACTION_PARSER,
                content=(
                    f'invalid action name {exp.response.action_name}. Action name should be one of the following:\n'
                    f"{action_names}"
                ),
                feedback_status=FEEDBACK_WRONG
            )
            # if parsing action failed, stop rest of the feedback funcs
            return [fb], FEEDBACK_FUNC_STOP

        # ========================== parsing action ==========================
        if 'Call tactic' in parsed_action_type.action_name:
            # parse name and subproblem
            oind, subproblem = None, None
            for okw in self.fuzz_kw_ls[0]:
                for pkw in self.fuzz_kw_ls[1]:
                    res = split_string_with_separators(exp.response.action_output, [okw, pkw])
                    oind, subproblem = [e[-1].strip() if all_exists(e[-1]) else None for e in res]
                    if all_exists(oind, subproblem):
                        break
                if all_exists(oind, subproblem):
                    break

            # fuzzy match of keywords, it's mostly llama3 having troubles with this
            if all_exists(subproblem) and (oind is None):
                for pkw in self.fuzz_kw_ls[1]:
                    oind_str = exp.response.action_output.split(pkw)[0].strip()
                    if all_exists(re.match(r'\d', oind_str)):
                        oind = oind_str
                        break

            if not all_exists(oind, subproblem):
                return [Feedback(
                    observer=ACTION_PARSER,
                    content=(
                        f'invalid format. Please respond with the following format'
                        f'\n"{ROUTING_RESP_TEMPLATE}"'
                    ),
                    feedback_status=FEEDBACK_WRONG
                )], FEEDBACK_FUNC_STOP

            assert all_exists(answer_set)
            if oind not in answer_set:
                answer_set_str = '\n'.join(e for e in answer_set)
                return [Feedback(
                    observer=ACTION_PARSER,
                    content=(
                        f'invalid option index. The index should be one of the following:\n'
                        f"{answer_set_str}"
                    ),
                    feedback_status=FEEDBACK_WRONG
                )], FEEDBACK_FUNC_STOP

            tactic_name = self.callname2tacticname[parsed_action_type.action_name]

            # NOTE we store subproblem traj in metadata, should think of a better way later
            subtraj_info = {
                'exp_ind': len(trajectory.exp_ls) if exp_ind == -1 else exp_ind,
                'tactic_name': tactic_name,
                'subproblem': subproblem,
                'oind': int(oind) - 1
            }
            if trajectory.metadata is None:
                trajectory.metadata = {'subtrajs': []}
            elif 'subtrajs' not in trajectory.metadata:
                trajectory.metadata['subtrajs'] = []
            trajectory.metadata['subtrajs'].append(subtraj_info)

            return [Feedback(
                observer=ACTION_PARSER,
                content=f'Solving subproblem with tactic {tactic_name}',
                feedback_status=FEEDBACK_OK
            )], FEEDBACK_FUNC_OK

        elif parsed_action_type.action_name == 'Aggregate and answer':
            if not answer_check:
                return [Feedback(
                    observer=ACTION_PARSER,
                    content='Answer not checked',
                    feedback_status=FEEDBACK_SOLVED
                )], FEEDBACK_FUNC_OK

            ans = exp.response.action_output.strip()
            assert all_exists(gt_answer, answer_set)

            # if ans not in answer_set
            if ans not in answer_set:
                answer_set_str = '\n'.join(e for e in answer_set)
                return [Feedback(
                    observer=ACTION_PARSER,
                    content=(
                        f'invalid answer\n\nThe output should be one of the following:\n'
                        f"{answer_set_str}"
                    ),
                    feedback_status=FEEDBACK_WRONG
                )], FEEDBACK_FUNC_STOP

            if isinstance(gt_answer, Callable):
                iscorrect = gt_answer(ans)
            else:
                iscorrect = gt_answer == ans

            # if ans is correct
            if iscorrect:
                return [Feedback(
                    observer=ACTION_PARSER,
                    content='Answer correct',
                    feedback_status=FEEDBACK_SOLVED
                )], FEEDBACK_FUNC_OK
            # if ans is incorrect
            else:
                return [Feedback(
                    observer=ACTION_PARSER,
                    content='Answer incorrect',
                    feedback_status=FEEDBACK_WRONG
                )], FEEDBACK_FUNC_OK

        else:
            raise ValueError('you should never reach here unless you forget to deal with this '
                             f'{parsed_action_type.action_name} action above')
