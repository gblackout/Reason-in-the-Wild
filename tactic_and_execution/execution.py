from typing import Optional, List, Callable, Dict
from .consts import *
from utils import all_exists
from .thought_action_observation_data_structure import Feedback, Response, RESP_TEMPLATE, ExpTuple, Trajectory
from functools import partial


class Execution:

    def __init__(self):
        self.feedback_func_ls = []

    def register_feedback_funcs(self, *funcs):
        for func in funcs:
            self.feedback_func_ls.append(func)

    def run_feedback_funcs(
            self,
            trajectory: Trajectory,
            parser_func: Optional[Callable] = None,
            extra_feedback_funcs: Optional[List[Callable]] = None
    ):
        entire_feedback_ls = []
        extra_feedback_funcs = extra_feedback_funcs if all_exists(extra_feedback_funcs) else []
        for func in [parser_func] + self.feedback_func_ls + extra_feedback_funcs:
            if func is None:
                continue
            feedback_ls, feedback_func_status = func(trajectory)
            if all_exists(feedback_ls):
                entire_feedback_ls.extend(feedback_ls)
            if feedback_func_status == FEEDBACK_FUNC_STOP:
                break
        return entire_feedback_ls if len(entire_feedback_ls) > 0 else None

    def default_exec_status_func(self, trajectory: Trajectory):
        """
            assign exec_status to the last exp of the traj, and return it
            default behavior:
            consider the last exp of traj: the exp has multiple feedbacks; use the most lazy feedback to determine the
            exec status
        :param trajectory:
        :return:
        """
        if trajectory.exp_ls is None:
            return EXEC_EMPTY

        exp = trajectory.exp_ls[-1]
        if exp.observation is None:
            return EXEC_EMPTY

        feedback_ls = exp.observation
        has_wrong = False
        for fb in feedback_ls:
            if fb.feedback_status == FEEDBACK_SOLVED:
                exp.exec_status = EXEC_SOLVED
                return EXEC_SOLVED
            elif fb.feedback_status == FEEDBACK_DEEMED_FAILED:
                exp.exec_status = EXEC_DEEMED_FAILED
                return EXEC_DEEMED_FAILED
            elif fb.feedback_status == FEEDBACK_WRONG:
                has_wrong = True

        exp.exec_status = EXEC_WRONG if has_wrong else EXEC_OK
        return exp.exec_status

    def default_termination_func(self, trajectory: Trajectory):
        return (
                (trajectory.cur_status() == EXEC_SOLVED) or
                (trajectory.cur_status() == EXEC_DEEMED_FAILED)
        )

    def parse_resp(
            self,
            trajectory: Trajectory,
            messages: Optional[List] = None,
            resp_str: Optional[str] = None,
            fuzzy_parse: bool = False,
            dup_detec: bool = True
    ):
        """
            this is a special feedback func that (1) runs before every other feedback funcs; (2) parses resp_str; (3)
            creates Response and ExpTuple objects from resp_str; (4) could stop running other funcs if parsing failed.
        :param messages:
        :param resp_str:
        :return:
        """
        if all_exists(messages):
            resp_msg = messages[-1]
            assert resp_msg["role"] == "assistant", 'the last message is not from assistant'
            resp_str = resp_msg['content']
        elif all_exists(resp_str):
            pass
        else:
            raise ValueError('at least give me one arg to parse from')

        # create exp tuple
        resp = Response.from_resp_str(resp_str, fuzzy_parse=fuzzy_parse)
        exp = ExpTuple(response=resp)
        trajectory.add_exp(exp)

        missing_fields = []
        if resp.thought is None:
            missing_fields.append('### Thought')
        if resp.action_name is None:
            missing_fields.append('## Name')
        if resp.action_input is None:
            missing_fields.append('## Input')
        if resp.action_output is None:
            missing_fields.append('## Output')

        if len(missing_fields) > 0:
            missing_field_str = '\n'.join(missing_fields)
            fb = Feedback(
                observer=RESPONSE_PARSER,
                content=(
                    f'invalid response format: the following fields are missing {missing_field_str}.\n'
                    f'Please respond with the following format\n"{RESP_TEMPLATE}"'
                ),
                feedback_status=FEEDBACK_WRONG
            )

            return [fb], FEEDBACK_FUNC_STOP

        # make sure only one such resp exists in output--sometimes, opus generates multiple of them in one go
        # since split_string_with_separators finds the first occurrence of all fields, the extra field will always be
        # in resp.action_output
        tmp_resp = Response.from_resp_str(resp.action_output)
        if (
            all_exists(tmp_resp.thought) or all_exists(tmp_resp.action_name) or
            all_exists(tmp_resp.action_input) or all_exists(tmp_resp.action_output)
        ):
            fb = Feedback(
                observer=RESPONSE_PARSER,
                content=(
                    f'invalid response format: multiple responses detected.\n'
                    f'Please respond with a single \n"{RESP_TEMPLATE}"'
                ),
                feedback_status=FEEDBACK_WRONG
            )

            return [fb], FEEDBACK_FUNC_STOP

        return None, FEEDBACK_FUNC_OK

    def check_should_terminate(
            self,
            trajectory: Trajectory,
            max_n_requests: int = 8,
            max_n_consecutive_wrong_actions: int = 3,
            max_n_wrong_actions: int = 10,
            termination_func: Optional[Callable] = None,
    ):

        reached_max_n_requests = (len(trajectory.exp_ls) >= max_n_requests) if all_exists(trajectory.exp_ls) else False
        reached_max_n_cwa = trajectory.max_consecutive_wrong_actions() >= max_n_consecutive_wrong_actions
        reached_max_n_wa = trajectory.total_wrong_actions() >= max_n_wrong_actions
        term_func = termination_func if all_exists(termination_func) else self.default_termination_func
        term_func_res = term_func(trajectory)

        should_terminate = (
                reached_max_n_requests or
                reached_max_n_cwa or
                reached_max_n_wa or
                term_func_res
        )

        return should_terminate

    def req_step(
            self,
            trajectory: Trajectory,
            request_func: Optional[Callable] = None,
            new_messages: Optional[Dict] = None,
            exec_status_func: Optional[Callable] = None,
            extra_feedback_funcs: Optional[List[Callable]] = None,
            fuzzy_parse: bool = False,
    ):

        exec_status_func = exec_status_func if all_exists(exec_status_func) else self.default_exec_status_func

        # new_messages is None is typically used in default sync traj gen
        # new_messages not None is typically used in batch async traj gen
        if new_messages is None:
            messages = trajectory.to_messages()
            assert all_exists(request_func), 'if you want me to gen new msg myself, give me request_func'
            new_messages = request_func(messages=messages)

            if new_messages is None:
                return None, None, TRAJ_REQUEST_FAILED

        # run all registered feedback funcs
        parser_func = partial(self.parse_resp, messages=new_messages, fuzzy_parse=fuzzy_parse)
        feedback_ls = self.run_feedback_funcs(
            trajectory,
            parser_func=parser_func,
            extra_feedback_funcs=extra_feedback_funcs
        )
        exp = trajectory.exp_ls[-1]
        exp.observation = feedback_ls  # NOTE this assumes observation not exists

        # run exec status func to determine status
        exec_status_func(trajectory)

        return new_messages, exp, TRAJ_OK

    def step(
            self,
            request_func: Callable,
            trajectory: Trajectory,
            max_n_requests: int = 8,
            max_n_consecutive_wrong_actions: int = 3,
            max_n_wrong_actions: int = 10,
            termination_func: Optional[Callable] = None,
            exec_status_func: Optional[Callable] = None,
            extra_feedback_funcs: Optional[List[Callable]] = None,
            fuzzy_parse: bool = False
    ):

        should_terminate = self.check_should_terminate(
            trajectory,
            max_n_requests,
            max_n_consecutive_wrong_actions,
            max_n_wrong_actions,
            termination_func,
        )

        if should_terminate:
            return None, None, TRAJ_TERMINATED

        return self.req_step(
            trajectory,
            request_func,
            new_messages=None,
            exec_status_func=exec_status_func,
            extra_feedback_funcs=extra_feedback_funcs,
            fuzzy_parse=fuzzy_parse
        )

