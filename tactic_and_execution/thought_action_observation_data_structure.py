import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Union
from utils import split_string_with_separators, all_exists
import os
from .consts import *
from itertools import product


RESP_TEMPLATE = """
### Thought
<your thought>

### Action
## Name
<exact name of the action>
## Input
<input of the action>
## Output
<output of the action>
"""


@dataclass
class ActionType:
    action_name: Optional[str] = None
    action_input: Optional[str] = None
    action_functionality: Optional = None
    action_output: Optional[str] = None

    @staticmethod
    def from_str(action_def_str):
        res = split_string_with_separators(action_def_str, [
            '#A#',
            '- Input:',
            '- Functionality:',
            '- Output:'
        ])

        an, ai, af, ao = [e[-1].strip() for e in res]
        assert all_exists(an, ai, af, ao), f'cannot parse action from str {action_def_str}'

        return ActionType(action_name=an, action_input=ai, action_functionality=af, action_output=ao)

    def __str__(self):
        return '\n'.join([
            f"#A# {self.action_name}",
            f"- Input: {self.action_input}",
            f"- Functionality: {self.action_functionality}",
            f"- Output: {self.action_output}",
        ])

    def to_dict(self):
        return str(self)

    @staticmethod
    def from_json(js):
        return ActionType.from_str(js) if all_exists(js) else None


class ActionSpace:

    def __init__(self, action_dict, space_instruction):
        self.action_dict = action_dict
        self.action_ls = [v for k, v in action_dict.items()]
        self.space_instruction = space_instruction

    def __getitem__(self, item):
        return self.action_dict[item]

    @staticmethod
    def from_doc(doc: str):
        """
            Construct the ActionSpace from a doc, which could be a tactic doc, the doc should have a section
            starting with **Action space** and ended with the definition of the last action and nothing else
        :param doc: the str, or the path to the str
        :return:
        """

        if os.path.isfile(doc) and doc.endswith('txt'):
            with open(doc, 'r') as f:
                doc_str = f.read()
        else:
            doc_str = doc

        space_start = '**Action space**'
        assert space_start in doc_str, f'cannot find {space_start} in the tactic doc'
        parts = doc_str.split(space_start)
        assert len(parts) == 2, f'there are {len(parts)} {space_start} found the tactic doc'

        space_str = parts[-1].strip()

        parts = space_str.split('#A#')
        space_instruction = parts[0].strip()

        action_dict = {}
        for part in parts[1:]:
            action_str = '#A#' + part
            action = ActionType.from_str(action_str)
            action_dict[action.action_name] = action

        return ActionSpace(action_dict=action_dict, space_instruction=space_instruction)

    def __str__(self):

        action_ls_str = '\n\n'.join([str(action) for action in self.action_ls])

        return ''.join([
            '**Action space**\n',
            f"{self.space_instruction}\n\n",
        ]) + action_ls_str

    def to_doc(self, doc_path):
        with open(doc_path, 'w') as f:
            f.write(str(self))


fuzzy_parse_kws_ls = [
    ['### Thought', '### Thoughts', '## Thought', '## Thoughts', '# Thought', '# Thoughts', '# Action'],
    ['### Action', '## Action', '# Action'],
    ['## Name', '### Name', '# Name'],
    ['## Input', '### Input', '# Input'],
    ['## Output', '### Output', '# Output'],
]


@dataclass
class Response:
    raw: Optional[str] = None
    thought: Optional[str] = None
    action_name: Optional[str] = None
    action_input: Optional[str] = None
    action_output: Optional[str] = None

    @staticmethod
    def from_resp_str(resp_str, fuzzy_parse: bool = False):

        # if fuzzy_parse is on, we try to match with fuzzy_parse_kws_ls, if this failed, then we back to default
        if fuzzy_parse:
            for t_kw, a_kw, an_kw, ai_kw, ao_kw in product(*fuzzy_parse_kws_ls):
                res = split_string_with_separators(resp_str, [t_kw, a_kw, an_kw, ai_kw, ao_kw])
                t, _, an, ai, ao = [e[-1].strip() if all_exists(e[-1]) else None for e in res]
                if all_exists(t, an, ai, ao):
                    return Response(raw=resp_str, thought=t, action_name=an, action_input=ai, action_output=ao)

        res = split_string_with_separators(resp_str, [
            '### Thought',
            '### Action',
            '## Name',
            '## Input',
            '## Output'
        ])
        t, _, an, ai, ao = [e[-1].strip() if all_exists(e[-1]) else None for e in res]
        return Response(raw=resp_str, thought=t, action_name=an, action_input=ai, action_output=ao)

    def __str__(self):
        return '\n'.join([
            '### Thought',
            f"{self.thought}",
            '### Action',
            '## Name',
            f"{self.action_name}",
            '## Input',
            f"{self.action_input}",
            '## Output',
            f"{self.action_output}"
        ])

    def to_dict(self):
        return {
            'raw': self.raw,
            'thought': self.thought,
            'action_name': self.action_name,
            'action_input': self.action_input,
            'action_output': self.action_output
        }

    @staticmethod
    def from_json(js):
        return Response(
            raw=js['raw'],
            thought=js['thought'],
            action_name=js['action_name'],
            action_input=js['action_input'],
            action_output=js['action_output']
        ) if all_exists(js) else None


@dataclass
class Feedback:
    observer: Optional[str] = None
    content: Optional[str] = None
    feedback_status: Optional[str] = None

    def __str__(self):
        return '\n'.join([
            f'# Observer: {self.observer}',
            f'# Feedback status: {self.feedback_status}',
            '# Content:',
            f'{self.content}'
        ])

    def to_dict(self):
        return {
            'observer': self.observer,
            'content': self.content,
            'feedback_status': self.feedback_status
        }

    @staticmethod
    def from_json(js):
        return Feedback(
            observer=js['observer'],
            content=js['content'],
            feedback_status=js['feedback_status']
        ) if all_exists(js) else None


@dataclass
class ExpTuple:
    response: Optional[Response] = None
    action_type: Optional[ActionType] = None
    observation: Optional[List[Feedback]] = None
    exec_status: Optional[str] = None

    def __str__(self):
        observation_str = ('\n\n'.join(str(fb) for fb in self.observation)) if all_exists(self.observation) else None
        return '\n\n'.join([
            f'=== status: {self.exec_status}',
            '=== response ===',
            f'{self.response.raw if all_exists(self.response) else None}',
            '=== action type ===',
            f'{str(self.action_type) if all_exists(self.action_type) else None}',
            '=== observations ===',
            f'{observation_str}'
        ])

    def nice_str(self):
        observation_str = ('\n\n'.join(str(fb) for fb in self.observation)) if all_exists(self.observation) else None
        return '\n\n'.join([
            f'=== response === status: {self.exec_status}',
            f'\033[96m{self.response.raw if all_exists(self.response) else None}\033[00m',
            '=== observations ===',
            f'{observation_str}'
        ])

    def to_dict(self):
        return {
            'response': self.response.to_dict() if all_exists(self.response) else None,
            'action_type': self.action_type.to_dict() if all_exists(self.action_type) else None,
            'observation': [fb.to_dict() for fb in self.observation] if all_exists(self.observation) else None,
            'exec_status': self.exec_status
        }

    @staticmethod
    def from_json(js):
        return ExpTuple(
            response=Response.from_json(js['response']),
            action_type=ActionType.from_json(js['action_type']),
            observation=[Feedback.from_json(e) for e in js['observation']] if ('observation' in js) else None,
            exec_status=js['exec_status']
        )


@dataclass
class Trajectory:
    init_prompt: Optional[str] = None
    exp_ls: Optional[List[ExpTuple]] = None
    metadata: Optional[Dict] = None

    def add_exp(self, exp: ExpTuple):
        if all_exists(self.exp_ls):
            self.exp_ls.append(exp)
        else:
            self.exp_ls = [exp]

    def to_messages(self, use_raw=True):

        messages = [
            {'role': 'user', 'content': self.init_prompt}
        ]

        if all_exists(self.exp_ls):
            for exp in self.exp_ls:
                model_resp = exp.response.raw if use_raw else str(exp.response)
                messages.append(
                    {'role': 'assistant', 'content': model_resp}
                )

                if all_exists(exp.observation):
                    feedback_str = '\n'.join(str(fb) for fb in exp.observation)
                    messages.append(
                        {'role': 'user', 'content': feedback_str}
                    )

        return messages

    def cur_status(self):
        return self.exp_ls[-1].exec_status if all_exists(self.exp_ls) else EXEC_EMPTY

    def show_traj_status(self):
        return [
            (exp.action_type.action_name if all_exists(exp.action_type) else None, exp.exec_status)
            for exp in self.exp_ls
        ] if all_exists(self.exp_ls) else []

    def total_wrong_actions(self):
        return len([exp.exec_status == EXEC_WRONG for exp in self.exp_ls]) if all_exists(self.exp_ls) else 0

    def max_consecutive_wrong_actions(self):
        if self.exp_ls is None:
            return 0
        cur_n, max_n = 0, 0
        isconsecutive = False
        for exp in self.exp_ls:
            if exp.exec_status == EXEC_WRONG:
                cur_n = (cur_n + 1) if isconsecutive else 1
                isconsecutive = True
            else:
                isconsecutive = False
            max_n = max(max_n, cur_n)
        return max_n

    def __str__(self):
        traj_str = '\n\n######\n\n'.join(str(exp) for exp in self.exp_ls) if all_exists(self.exp_ls) else None
        return '\n\n'.join([
            '=== init prompt ===',
            f'{self.init_prompt}',
            '=== trajectory ===',
            f'{traj_str}'
        ])

    def nice_str(self, include_init_prompt=False):
        # if include_init_prompt is False, we only include the orig question, assuming tactic_exec_prompt is used
        # then the orig question can be extracted from prompt this way
        parts = split_string_with_separators(self.init_prompt, ['=== Question', '=== Tactic'])
        init_str = self.init_prompt if include_init_prompt else parts[0][-1]

        traj_str = '\n\n######\n\n'.join(exp.nice_str() for exp in self.exp_ls) if all_exists(self.exp_ls) else None
        return '\n\n'.join([
            '=== init prompt ===',
            f'{init_str}',
            '=== trajectory ===',
            f'{traj_str}'
        ])

    def to_dict(self):
        return {
            'metadata': self.metadata,
            'init_prompt': self.init_prompt,
            'exp_ls': [exp.to_dict() for exp in self.exp_ls] if all_exists(self.exp_ls) else None
        }

    @staticmethod
    def from_json(js: Union[Dict, str]):
        if isinstance(js, str):
            with open(js, 'r') as f:
                js = json.load(f)

        has_exp_ls = False
        if 'exp_ls' in js:
            if all_exists(js['exp_ls']):
                has_exp_ls = True

        return Trajectory(
            metadata=js['metadata'] if 'metadata' in js else None,
            init_prompt=js['init_prompt'],
            exp_ls=[ExpTuple.from_json(e) for e in js['exp_ls']] if has_exp_ls else None
        )
