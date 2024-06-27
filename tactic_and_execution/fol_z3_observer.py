from .observer import BaseObserver
from .thought_action_observation_data_structure import Trajectory, Feedback
from .consts import *
from utils import all_exists, split_string_with_separators
from utils.python_code_parser import find_func, run_code
from typing import Callable


z3_code_template = """
import z3
from z3 import *

def check_model(solver):
    res = solver.check()
    if res == z3.sat:
        return 'sat'
    elif res == z3.unsat:
        return 'unsat'
    else:
        return 'unsolvable'

def check_constraint(solver, c):
    pos_res = solver.check(c)
    neg_res = solver.check(z3.Not(c))

    if (pos_res == z3.sat) and (neg_res == z3.unsat):
        return 'Agree'
    elif (pos_res == z3.unsat) and (neg_res == z3.sat):
        return 'Contradict'
    elif (pos_res == z3.unknown) or (neg_res == z3.unknown):
        return 'unsolvable'
    else:
        return 'Uncertain'

{generated_main}

res = main()
if res is not None:
    print(res)
"""

MAIN_FUNC_NAME = 'main'


class FOLZ3Observer(BaseObserver):

    @staticmethod
    def check_answer(gt, pred, *args, **kwargs):
        return gt == pred

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
        if parsed_action_type.action_name == 'Tactic check':
            if 'Tactic Good' in exp.response.action_output:
                fb = Feedback(
                    observer=ECHO,
                    content='OK!',
                    feedback_status=FEEDBACK_OK,
                )
                return [fb], FEEDBACK_FUNC_OK
            elif 'Tactic Bad' in exp.response.action_output:
                fb = Feedback(
                    observer=ECHO,
                    content='Tactic Bad',
                    feedback_status=FEEDBACK_DEEMED_FAILED,
                )
                # i think deemed failed should not stop rest feedbacks, we will see
                return [fb], FEEDBACK_FUNC_OK
            # the case it does not give right answer
            else:
                return [Feedback(
                    observer=ACTION_PARSER,
                    content=(
                        f'invalid output {exp.response.action_output}. The output should be one of the following:\n'
                        f"Tactic Good\nTactic Bad"
                    ),
                    feedback_status=FEEDBACK_WRONG
                )], FEEDBACK_FUNC_STOP

        elif parsed_action_type.action_name == 'Plan':
            fb = Feedback(
                observer=ECHO,
                content='OK!',
                feedback_status=FEEDBACK_OK,
            )
            return [fb], FEEDBACK_FUNC_OK

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
                        f'invalid answer\n{ans}\n\nThe output should be one of the following:\n'
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

        elif (
                (parsed_action_type.action_name == 'Build FOL model') or
                (parsed_action_type.action_name == 'Revise code')
        ):

            ans = exp.response.action_output.strip()
            # find the code section
            res = split_string_with_separators(ans, ['```python', '```'])
            code_str = res[0][-1]
            if code_str is None:
                return [Feedback(
                    observer=ACTION_PARSER,
                    content='cannot find python code. Please wrap the code as\n```python\n<code>\n```',
                    feedback_status=FEEDBACK_WRONG
                )], FEEDBACK_FUNC_STOP

            # parse the python code
            target_func, err_msg, all_func_ls = find_func(code_str, MAIN_FUNC_NAME)
            # if parsing failed
            if all_exists(err_msg):
                return [Feedback(
                    observer=PYTHON_INTERPRETER,
                    content=f'cannot parse python code:\n{err_msg}',
                    feedback_status=FEEDBACK_WRONG
                )], FEEDBACK_FUNC_STOP
            # if cannot find main func
            if target_func is None:
                return [Feedback(
                    observer=ACTION_PARSER,
                    content=f'cannot find {MAIN_FUNC_NAME} func in the code',
                    feedback_status=FEEDBACK_WRONG
                )], FEEDBACK_FUNC_STOP

            # check if other funcs are returned, and tell model not to do so
            redundant_func_names = [func_name for func_name, func in all_func_ls if func_name != MAIN_FUNC_NAME]
            if len(redundant_func_names) > 0:
                feedbacks.append(Feedback(
                    observer=ACTION_PARSER,
                    content=(
                        'warning: functions other than main() are detected:\n'
                        f"{','.join(redundant_func_names)}\n"
                        'please only return main() as other functions are discarded'
                    ),
                    feedback_status=FEEDBACK_WARNING
                ))

            # now run the code
            complete_code_str = z3_code_template.format(generated_main=target_func)
            code_res = run_code(complete_code_str)

            # determine feedback with exit code
            if code_res.returncode == 0:
                run_feedback_status = FEEDBACK_OK
                res_str = f'stdout:\n{code_res.stdout}'
            else:
                run_feedback_status = FEEDBACK_WRONG
                res_str = f'stdout:\n{code_res.stdout}\nstderr:\n{code_res.stderr}'

            feedbacks.append(Feedback(
                observer=PYTHON_INTERPRETER,
                content=res_str,
                feedback_status=run_feedback_status
            ))

            return feedbacks, FEEDBACK_FUNC_OK

        else:
            raise ValueError('you should never reach here unless you forget to deal with this '
                             f'{parsed_action_type.action_name} action above')
