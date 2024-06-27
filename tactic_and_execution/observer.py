from .thought_action_observation_data_structure import ExpTuple, ActionSpace


class BaseObserver:

    def __init__(self, tactic_doc: str):
        self.action_space = ActionSpace.from_doc(tactic_doc)

    def parse_action_type(self, exp: ExpTuple):
        resp = exp.response
        # if directly match, then simple
        if resp.action_name in self.action_space.action_dict:
            exp.action_type = self.action_space.action_dict[resp.action_name]
            return exp.action_type
        # if not, try lowercase
        else:
            for name in self.action_space.action_dict:
                if name.lower() in resp.action_name.lower():
                    exp.action_type = self.action_space.action_dict[name]
                    return exp.action_type
            # if still not found, then failed
            return None

    def give_feedback(self, messages, **kwargs):
        raise NotImplementedError

    @staticmethod
    def check_answer(*args, **kwargs):
        raise NotImplementedError
