import json
import time
from openai import OpenAI
from utils.misc import wrap_function_with_timeout, make_parent_dirs
from typing import Optional, Dict, List, Union, Callable
from utils import all_exists
import os
from functools import partial
import logging
import google.generativeai as genai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import anthropic
from copy import deepcopy
from dataclasses import dataclass
from openai.types import FileObject as OpenAIFile
from openai.types import Batch as OpenAIBatch
from collections import defaultdict
import threading
import cohere


def convert_message_format(orig_messages, to_format):

    messages = deepcopy(orig_messages)

    if to_format == 'gpt':
        for m in messages:
            if m['role'] == 'model':
                m['role'] = 'assistant'  # others uses assistant
            if m['role'] == 'assistant':
                pass
            elif m['role'] == 'user':
                pass  # both format shares the same name of user
            else:
                raise ValueError(f"invalid role name {m['role']} in message")

            if 'parts' in m:
                m['content'] = m['parts'][0]
                del m['parts']
            elif 'content' in m:
                pass # already in others format
            else:
                raise ValueError(f'I cant find parts or content in this message {m}')

    elif to_format == 'gemini':
        for m in messages:
            if m['role'] == 'assistant':
                m['role'] = 'model' # google uses model
            if m['role'] == 'model':
                pass
            elif m['role'] == 'user':
                pass # both format shares the same name of user
            else:
                raise ValueError(f"invalid role name {m['role']} in message")

            if 'parts' in m:
                pass # already the google format
            elif 'content' in m:
                m['parts'] = [m['content']] # convert content to parts
                del m['content']
            else:
                raise ValueError(f'I cant find parts or content in this message {m}')

    elif to_format == 'cohere':
        for m in messages:
            if m['role'] == 'assistant':
                m['role'] = 'CHATBOT'
            elif m['role'] == 'user':
                m['role'] = 'USER'

            m['message'] = m['content']
            del m['content']

        # NOTE this is assuming the last message is from user
        user_message = messages.pop(-1)
        chat_history = messages
        user_msg = user_message['message']

        return chat_history, user_msg

    else:
        raise ValueError(f'{to_format} not supported')

    return messages


class LLMRequestManager:

    model_gpt4: str = 'gpt-4'
    model_gpt4_turbo: str = 'gpt-4-1106-preview'
    model_gpt4_turbo_0409: str = 'gpt-4-turbo-2024-04-09'
    model_gpt4o: str = 'gpt-4o'
    model_gpt35: str = 'gpt-3.5-turbo'
    model_gemini_pro: str = 'gemini-1.0-pro'
    model_gemini15_pro: str = 'gemini-1.5-pro'
    model_mistral7b: str = 'mistral-tiny'
    model_mixtral: str = 'mistral-small'
    model_mistral_medium: str = 'mistral-medium'
    model_mistral_large: str = 'mistral-large-latest'
    model_claude_opus: str = 'claude-3-opus-20240229'
    model_claude_sonnet: str = 'claude-3-sonnet-20240229'
    model_command_r_plus: str = 'command-r-plus'

    def __init__(self, api_key_file: str):
        """
        :param api_key_file:
            the path to the key file with every line being <type>: <key>
            where <type> should be one of the following: openai, gemini, mistral, anthropic, cohere
        """
        self.available_request_funcs = set()
        assert os.path.isfile(api_key_file)

        with open(api_key_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                parts = list(map(lambda x:x.strip(), line.split(':')))
                assert len(parts) == 2, f'api key file parsing failed on {line}\n make sure the format is <type>: <key>'
                llm_type, llm_key = parts[0], parts[1]
                if llm_type == 'openai':
                    self.openai_client = OpenAI(api_key=llm_key)
                    self.available_request_funcs.add(self.gpt_request)
                elif llm_type == 'gemini':
                    genai.configure(api_key=llm_key)
                    self.available_request_funcs.add(self.gemini_request)
                elif llm_type == 'mistral':
                    self.mistral_client = MistralClient(api_key=llm_key)
                    self.available_request_funcs.add(self.mistral_request)
                elif llm_type == 'anthropic':
                    self.anthropic_client = anthropic.Anthropic(api_key=llm_key)
                    self.available_request_funcs.add(self.claude_request)
                elif llm_type == 'cohere':
                    self.cohere_client = cohere.Client(llm_key)
                    self.available_request_funcs.add(self.cohere_request)
                else:
                    raise ValueError(f'unknown type {llm_type}')

    def cohere_request(
            self,
            model,
            messages: Optional[List] = None,
            resp_split_func: Optional[Callable] = None,
            return_messages: bool = False
    ):

        chat_history, user_msg = convert_message_format(messages, 'cohere')

        try:
            chat = self.cohere_client.chat(
                chat_history=chat_history,
                message=user_msg,
                model=model
            )
            resp_str = chat.text
        except Exception as e:
            logging.exception('something wrong with the request:\n' + str(e))
            return None

        if all_exists(resp_split_func):
            assert not return_messages, 'return_messages not supported for resp_split_func'
            return resp_split_func(resp_str)

        if return_messages:
            messages.append({"role": "assistant", "content": resp_str})
            return messages

        return resp_str

    def claude_request(
            self,
            model,
            messages: Optional[List] = None,
            resp_split_func: Optional[Callable] = None,
            return_messages: bool = False
    ):

        try:
            message = self.anthropic_client.messages.create(
                model=model,
                max_tokens=2048, # ad-hoc
                messages=messages
            )
            resp_str = message.content[0].text
        except Exception as e:
            logging.exception('something wrong with the request:\n' + str(e))
            return None

        if all_exists(resp_split_func):
            assert not return_messages, 'return_messages not supported for resp_split_func'
            return resp_split_func(resp_str)

        if return_messages:
            messages.append({"role": "assistant", "content": resp_str})
            return messages

        return resp_str

    def mistral_request(
            self,
            model,
            messages,
            resp_split_func: Optional[Callable] = None,
            return_messages: bool = False
    ):

        mistral_messages = [
            ChatMessage(role=m['role'], content=m['content'])
            for m in messages
        ]

        try:
            response = self.mistral_client.chat(
                model=model,
                messages=mistral_messages
            )
            resp_str = response.choices[0].message.content
        except Exception as e:
            logging.exception('something wrong with the request:\n' + str(e))
            return None

        if all_exists(resp_split_func):
            assert not return_messages, 'return_messages not supported for resp_split_func'
            return resp_split_func(resp_str)

        if return_messages:
            messages.append({"role": "assistant", "content": resp_str})
            return messages

        return resp_str

    def gpt_request(
            self,
            model,
            input_prompt: Optional[str] = None,
            system_prompt: Optional[str] = None,
            messages: Optional[List] = None,
            resp_split_func: Optional[Callable] = None,
            return_messages: bool = False
    ):
        # the case where a single-round of prompt is given
        if all_exists(input_prompt, system_prompt):
            assert messages is None, "you already gave me input and system prompt, I can't take messages arg"
            if system_prompt is None:
                messages = [
                    {"role": "user", "content": input_prompt},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_prompt},
                ]
        # the case where a multi-round/custome prompt is given
        elif all_exists(messages):
            pass

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages
            )
            resp_str = response.choices[0].message.content
        except Exception as e:
            logging.exception('something wrong with the request:\n' + str(e))
            return None

        if all_exists(resp_split_func):
            assert not return_messages, 'return_messages not supported for resp_split_func'
            return resp_split_func(resp_str)

        if return_messages:
            messages.append({"role": "assistant", "content": resp_str})
            return messages

        return resp_str

    def gemini_request(
            self,
            model,
            messages: Optional[List] = None,
            resp_split_func: Optional[Callable] = None,
            return_messages: bool = False
    ):

        model = genai.GenerativeModel(model)

        # convert from gpt to google format
        to_send_messages = convert_message_format(messages, to_format='gemini')

        try:
            response = model.generate_content(to_send_messages)
            resp_str = response.text
        except Exception as e:
            logging.exception('something wrong with the request:\n' + str(e))
            return None

        if all_exists(resp_split_func):
            assert not return_messages, 'return_messages not supported for resp_split_func'
            return resp_split_func(resp_str)

        if return_messages:
            messages.append({"role": "assistant", "content": resp_str})
            return messages

        return resp_str

    def default_request(
            self,
            model: str,
            input_prompt: Optional[str] = None,
            system_prompt: Optional[str] = None,
            messages: Optional[List] = None,
            resp_split_func: Optional[Callable] = None,
            return_messages: Optional[bool] = None
    ):
        """

        Post a request to a GPT/Gemini model; either give me a single-round input: input_prompt + system_prompt, or
        your own messages list following the openai/gemini format.

        :param model:
        :param input_prompt:
        :param system_prompt:
            only used for gpt, and not needed for gemini
        :param messages:
        :param resp_split_func:
        :return:
        """
        # gpt
        if (
                model == LLMRequestManager.model_gpt35 or
                model == LLMRequestManager.model_gpt4 or
                model == LLMRequestManager.model_gpt4_turbo or
                model == LLMRequestManager.model_gpt4_turbo_0409 or
                model == LLMRequestManager.model_gpt4o
        ) and self.gpt_request in self.available_request_funcs:
            return self.gpt_request(
                model=model,
                input_prompt=input_prompt,
                system_prompt=system_prompt,
                messages=messages,
                resp_split_func=resp_split_func,
                return_messages=return_messages
            )
        # gemini
        elif (
                model == LLMRequestManager.model_gemini_pro or
                model == LLMRequestManager.model_gemini15_pro
        ) and self.gemini_request in self.available_request_funcs:
            return self.gemini_request(
                model=model,
                messages=messages,
                resp_split_func=resp_split_func,
                return_messages=return_messages
            )
        # mistral
        elif (
            model == LLMRequestManager.model_mistral7b or
            model == LLMRequestManager.model_mixtral or
            model == LLMRequestManager.model_mistral_medium or
            model == LLMRequestManager.model_mistral_large
        ) and self.mistral_request in self.available_request_funcs:
            return self.mistral_request(
                model=model,
                messages=messages,
                resp_split_func=resp_split_func,
                return_messages=return_messages
            )
        # claude
        elif (
            model == LLMRequestManager.model_claude_opus or
            model == LLMRequestManager.model_claude_sonnet
        ) and self.claude_request in self.available_request_funcs:
            return self.claude_request(
                model=model,
                messages=messages,
                resp_split_func=resp_split_func,
                return_messages=return_messages
            )
        # cohere
        elif (
                model == LLMRequestManager.model_command_r_plus
        ) and self.cohere_request in self.available_request_funcs:
            return self.cohere_request(
                model=model,
                messages=messages,
                resp_split_func=resp_split_func,
                return_messages=return_messages
            )
        else:
            raise ValueError(f'model {model} is either unknown; or it is not available because of missing api_key')

    def request_for_dataset(
            self,
            dataset: Union[str, List],
            resp_key: str,
            timeout: int = 300,
            prompt_prep_func: Optional[Callable] = None,
            resp_split_func: Optional[Callable] = None,
            n_retry: int = 3,
            tqdm: Optional[Callable] = None,
            verbose_func: Optional[Callable] = None,
            save_path: Optional[str] = None,
            model: str = 'gpt-3.5-turbo',
            save_every_nrequests: int = 10,
            filter_func: Optional[Callable] = None,
    ):
        """
            Post GPT requests for samples in a dataset.

        :param dataset:
        :param prompt_prep_func:
            given the sample in this dataset, you specify the func of how to make a prompt out of it, the output
            should be a named dict having either {'input_prompt': blabla, 'system_prompt': blabla} or
            {'messages': messages} where messages need to follow the openai format

        :param resp_key:
            I will put gpt response to sample[resp_key]
        :param timeout:
            timeout for the request
        :param resp_split_func:
        :param n_retry:
        :param tqdm:
        :param verbose_func:
            given ind, sample, and resp, you specify a func to print it
        :param save_path:
        :param model:
        :param save_every_nrequests:
        :param filter_func:
            given a sample, you specify a func to determine if you want to process it or not
        :return:
        """

        request_with_timeout = wrap_function_with_timeout(self.default_request, timeout)

        if isinstance(dataset, str):
            assert os.path.isfile(dataset) and dataset.endswith('json')
            with open(dataset, 'r') as f:
                dataset = json.load(f)
        assert isinstance(dataset, List)

        assert all_exists(save_path)
        make_parent_dirs(save_path)

        pbar = tqdm(dataset, leave=False) if all_exists(tqdm) else None
        update_bar = pbar.update if all_exists(tqdm) else lambda: None

        filter_func = filter_func if all_exists(filter_func) else lambda x: True
        _prompt_prep_func = prompt_prep_func if all_exists(prompt_prep_func) else lambda x: x['input']

        cnt = 0
        for ind, entry in enumerate(dataset):
            entry_is_valid = filter_func(entry)
            resp_exists = (resp_key in entry) and all_exists(entry[resp_key])
            should_request = entry_is_valid and (not resp_exists)

            if not should_request:
                update_bar()
                continue

            resp = None
            prompt_dict = prompt_prep_func(entry)
            for _ in range(n_retry):
                resp = request_with_timeout(
                    model=model,
                    resp_split_func=resp_split_func,
                    **prompt_dict
                )
                if all_exists(resp):
                    break

            if resp is None:
                logging.info(f'{ind} no response')
            else:
                entry[resp_key] = resp
                if all_exists(verbose_func):
                    verbose_func(ind, entry, resp)

            cnt += 1
            if cnt % save_every_nrequests == 0:
                with open(save_path, 'w') as f:
                    json.dump(dataset, f)

            update_bar()

        with open(save_path, 'w') as f:
            json.dump(dataset, f)


@dataclass
class BatchTracker:
    oai_f_ls: Optional[List[OpenAIFile]] = None
    oai_b_ls: Optional[List[OpenAIBatch]] = None
    ids_by_fn: Optional[Dict] = None
    fn_in_ls: Optional[List] = None
    fn_out_ls: Optional[List] = None
    total_n: Optional[int] = None


class BatchReqManager:

    def __init__(
            self,
            llm_manager: Optional[LLMRequestManager] = None,
            api_key_file: Optional[str] = None
    ):
        if all_exists(llm_manager):
            self.llm_manager = llm_manager
        elif all_exists(api_key_file):
            self.llm_manager = LLMRequestManager(api_key_file=api_key_file)
        else:
            raise ValueError('at least give me one arg here')

    @staticmethod
    def make_jsonl_entry(custom_id, model_name, messages):
        return {
            'custom_id': custom_id,
            'method': 'POST',
            "url": "/v1/chat/completions",
            "body": {
                'model': model_name,
                'messages': messages
            }
        }

    def upload_file(self, fn):
        if isinstance(fn, str):
            with open(fn, 'rb') as f:
                oai_f = self.llm_manager.openai_client.files.create(
                    file=f,
                    purpose='batch'
                )
                return oai_f
        else:
            oai_f = self.llm_manager.openai_client.files.create(
                file=fn,
                purpose='batch'
            )
            return oai_f

    def create_batch(self, oai_fid):
        oai_b = self.llm_manager.openai_client.batches.create(
            input_file_id=oai_fid,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return oai_b

    def upload_and_start_batch(self, fn: str):

        # split batch into mini-batches with the same model
        ids_by_fn = defaultdict(dict)
        fn_in_set = set()
        cnt = 0
        with open(fn, 'r') as f:
            for ind, line in enumerate(f):
                js = json.loads(line)
                new_fn, js_id = fn+'-'+js['body']['model'], js['custom_id']
                ids_by_fn[new_fn][js_id] = [ind, js]
                fn_in_set.add(new_fn)
                cnt += 1

        # save mini-batches to separate files and launch batches
        fn_in_ls = list(fn_in_set)
        oai_f_ls, oai_b_ls = [], []
        for new_fn in fn_in_ls:
            with open(new_fn, 'w') as f:
                for _, js in ids_by_fn[new_fn].values():
                    s = json.dumps(js)
                    f.write(s + '\n')

            oai_f = self.upload_file(new_fn)
            oai_b = self.create_batch(oai_f.id)
            oai_f_ls.append(oai_f)
            oai_b_ls.append(oai_b)

        tracker = BatchTracker(
            oai_f_ls=oai_f_ls,
            oai_b_ls=oai_b_ls,
            ids_by_fn=ids_by_fn,
            fn_in_ls=fn_in_ls,
            total_n=cnt
        )

        return tracker

    def watch_batch(
            self,
            oai_bid,
            check_every: int = 180,
            verbose: bool = True
    ):
        while True:
            oai_binfo = self.llm_manager.openai_client.batches.retrieve(oai_bid)
            if verbose:
                print(oai_binfo.status, oai_bid)
            if (
                    (oai_binfo.status == 'completed') or
                    (oai_binfo.status == 'failed') or
                    (oai_binfo.status == 'expired') or
                    (oai_binfo.status == 'cancelled')
            ):
                break
            time.sleep(check_every)

        oai_binfo = self.llm_manager.openai_client.batches.retrieve(oai_bid)
        return oai_binfo

    def download_batch(self, oai_bid, fp):
        oai_binfo = self.llm_manager.openai_client.batches.retrieve(oai_bid)
        assert oai_binfo.status == 'completed', f'cant download batch; batch status {oai_binfo.status}'
        oai_output_f = self.llm_manager.openai_client.files.content(oai_binfo.output_file_id)
        with open(fp, 'wb') as f:
            f.write(oai_output_f.content)

    def watch_and_download_batch(
            self,
            oai_bid,
            fp,
            check_every: int = 180,
            verbose: bool = True
    ):
        oai_binfo = self.watch_batch(oai_bid, check_every, verbose)
        if oai_binfo.status == 'completed':
            self.download_batch(oai_bid, fp)
            return 'completed'
        else:
            print(f'something wrong with batch process; batch status {oai_binfo.status}')
            return oai_binfo.status

    def watch_and_download_batches(
            self,
            tracker: BatchTracker,
            fp,
            check_every: int = 180,
            verbose: bool = True
    ):
        threads = []
        fn_out_ls = []
        for ind, oai_b in enumerate(tracker.oai_b_ls):
            new_fp = f'{fp}-{ind}'
            watch_download_one = partial(self.watch_and_download_batch, oai_b.id, new_fp, check_every, verbose)
            fn_out_ls.append(new_fp)
            t = threading.Thread(target=watch_download_one)
            threads.append(t)
            t.start()

        tracker.fn_out_ls = fn_out_ls

        for t in threads:
            t.join()

        if all([
            self.llm_manager.openai_client.batches.retrieve(oai_b.id).status == 'completed'
            for oai_b in tracker.oai_b_ls
        ]):
            return 'completed'
        else:
            return 'something wrong'

    def batch2messages(self, fn: Union[None, str, List] = None, tracker: Optional[BatchTracker] = None):

        assert not all_exists(fn, tracker), 'either give me fn or a tracker not both'

        if all_exists(fn):
            batch_resp = []
            if isinstance(fn, str):
                with open(fn, 'r') as f:
                    for line in f:
                        batch_resp.append(json.loads(line)['response']['body']['choices'][0]['message'])
                    return batch_resp
            else:
                for line in fn:
                    batch_resp.append(json.loads(line)['response']['body']['choices'][0]['message'])
                return batch_resp

        elif all_exists(tracker):
            batch_resp = [None for _ in range(tracker.total_n)]
            for new_fn_in, new_fn_out in zip(tracker.fn_in_ls, tracker.fn_out_ls):
                with open(new_fn_out, 'r') as f:
                    for line in f:
                        js = json.loads(line)
                        js_id = js['custom_id']
                        orig_ind = tracker.ids_by_fn[new_fn_in][js_id][0]
                        batch_resp[orig_ind] = js['response']['body']['choices'][0]['message']
            return batch_resp

        else:
            raise ValueError('give me at least one arg')