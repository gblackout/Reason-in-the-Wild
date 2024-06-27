from .observer import BaseObserver
from .thought_action_observation_data_structure import Trajectory, Feedback
from .consts import *
from utils import all_exists, split_string_with_separators
from utils.python_code_parser import run_code
from typing import Callable


class MathObserver(BaseObserver):

    @staticmethod
    def check_answer(gt, pred, *args, **kwargs):
        gt = gt.replace(',', '')
        try:
            pred = int(float(pred))
        except:
            pass
        return int(gt) == pred

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

        elif parsed_action_type.action_name == 'Reason':
            fb = Feedback(
                observer=ECHO,
                content='OK!',
                feedback_status=FEEDBACK_OK,
            )
            return [fb], FEEDBACK_FUNC_OK

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
            assert all_exists(gt_answer), 'you didnt give me gt_answer'

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
                    content=(
                        'Answer incorrect, make sure you return only the numerical result without '
                        'any explanation, description, and unit'
                    ),
                    feedback_status=FEEDBACK_WRONG
                )], FEEDBACK_FUNC_OK

        # for all other actions, we process it as python code
        elif (
            (parsed_action_type.action_name == 'Build math model') or
            (parsed_action_type.action_name == 'Revise code')
        ):
            ans = exp.response.action_output.strip()
            # find the code section
            res = split_string_with_separators(ans, ['```python', '```'])
            code_str = res[0][-1]

            if code_str is None:
                return [Feedback(
                    observer=PYTHON_INTERPRETER,
                    content=(
                        "No python code detected. Make sure you return a program of the form\n"
                        "```python\n<your code>\n```"
                    ),
                    feedback_status=FEEDBACK_WRONG
                )], FEEDBACK_FUNC_STOP

            # run the code
            code_res = run_code(code_str)

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
