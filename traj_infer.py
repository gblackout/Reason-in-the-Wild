from tactic_and_execution.thought_action_observation_data_structure import *
from typing import List, Dict
from tqdm import tqdm
from tactic_and_execution.traj_utils import traj2seq_simple
from functools import partial
from generate import MyLlamaTokenizer, simple_generate
from transformers import AutoModel
from utils.misc import load_model_and_peft
import fire
from traj_runner import TrajRunner
from utils.llm_requests import LLMRequestManager
from peft import PeftModel


def llama_req_func(
        traj: Trajectory,
        tk: MyLlamaTokenizer,
        model: AutoModel,
        messages: List[Dict],
        add_resp_head: bool = False,
        add_boseos_to_icl_prompt: bool = False,
        which_adapter: Optional[str] = None,
        is_hyb: bool = False
):
    # messages not used here, because I need to add eosbos before feeding to llama and remove them afterwards

    # convert traj to seq, the init traj has no exp_ls, so set no_exp_ok=True
    seq_ls = traj2seq_simple(traj, no_exp_ok=True, remove_hints=False, add_resp_head=add_resp_head,
                             add_boseos_to_icl_prompt=add_boseos_to_icl_prompt)
    inputs = tk(
        seq_ls,
        add_bos=False,
        add_eos=False,
        single_dim=False,
        my_token2llama_token=True,
        return_pt=True,
    )
    input_id_len = int(inputs['input_ids'].shape[1])

    # when a resp head (see traj2seq_simple for why) is added to seq_ls[-1], the actual input len is the seq_ls[:-1]
    if add_resp_head:
        inputs_no_head = tk(
            seq_ls[:-1],
            add_bos=False,
            add_eos=False,
            single_dim=False,
            my_token2llama_token=True,
            return_pt=True,
        )
        input_id_len = int(inputs_no_head['input_ids'].shape[1])

    isrouter = isinstance(model, PeftModel) and (which_adapter is None)
    istask = isinstance(model, PeftModel) and all_exists(which_adapter)
    # if its a task trajectory, we load the task adapter
    if istask:
        cur_adapter_name = model.active_adapters[0]
        model.set_adapter(which_adapter)

    if is_hyb:
        max_new_tokens = 512 if isrouter else 1024
    else:
        max_new_tokens = 1024
    resp_ids = simple_generate(
        model=model,
        input_ids=inputs['input_ids'],
        max_new_tokens=max_new_tokens,
    )

    resp_ids = resp_ids[input_id_len:]
    resp_str = tk.decode(resp_ids, skip_special_tokens=True)
    messages.append({"role": "assistant", "content": resp_str})

    # once the task generation is done, we restore the original adapter, that is the router adapter
    if istask:
        model.set_adapter(cur_adapter_name)

    return messages


def traj_infer(
        datapath: str,
        base_model: str,
        is_local: bool,
        save_key: str,
        save_path: str,
        hyb_run: bool,
        use_icl: bool,
        add_boseos_to_icl_prompt: bool, # llama setting
        skip_done: bool = False,
        peft_path: Optional[str] = None,
        router_peft_path: Optional[str] = None,
        use_icl_before_round: int = -1,
        use_baby_prompt_after_icl: bool = False,
        save_every: int = 10,
        req_wait: int = -1
):
    if is_local:
        tk = MyLlamaTokenizer(base_model)
        if all_exists(router_peft_path):
            print('router_peft_path detected, loading 2 adapters and set routing active')
            model = load_model_and_peft(
                base_model,
                load_in_8bit=True,
                peft_path=[(peft_path, 'default'), (router_peft_path, 'router')],
                use_flash_attn=True
            )
        else:
            model = load_model_and_peft(
                base_model,
                load_in_8bit=True,
                peft_path=peft_path,
                use_flash_attn=True
            )
        # the llama request func: set add_resp_head, to hint llama to always generate resp in an ad-hoc manner
        partial_request_func = partial(
            llama_req_func,
            tk=tk,
            model=model,
            add_resp_head=True,
            add_boseos_to_icl_prompt=add_boseos_to_icl_prompt
        )
    else:
        manager = LLMRequestManager('./llm-api-keys.txt')
        partial_request_func = partial(manager.default_request, model=base_model, return_messages=True)

    with open('data/icl_prompt_dict.json', 'r') as f:
        icl_prompt_dict = json.load(f)
    if not use_icl:
        for k in icl_prompt_dict:
            icl_prompt_dict[k] = ''

    runner = TrajRunner()

    def filter_skip_done(e, sd, sk):
        if sd:
            return not (sk in e)
        return True
    filter_func = partial(filter_skip_done, sd=skip_done, sk=save_key)

    runner.run_all_trajs(
        data=datapath,
        save_key=save_key,
        request_func=partial_request_func,
        filter_func=filter_func,
        use_icl_before_round=use_icl_before_round,
        max_n_requests=7,
        max_n_consecutive_wrong_actions=3,
        max_n_wrong_actions=10,
        verbose=True,
        give_traj_to_request_func=is_local,
        save_every=save_every,
        save_path=save_path,
        tqdm_func=tqdm,
        fuzzy_parse=True,
        stop_on_answer=True,
        use_baby_prompt_after_icl=use_baby_prompt_after_icl,
        hyb_run=hyb_run,
        icl_prompt_dict=icl_prompt_dict,
        sub_traj_init_kwargs={'use_icl_before_round': 2},
        sub_traj_run_kwargs={'give_traj_to_request_func': is_local},
        req_wait=req_wait,
        task_adapter_name='default' if all_exists(router_peft_path) else None
    )


if __name__ == '__main__':
    fire.Fire(traj_infer)

