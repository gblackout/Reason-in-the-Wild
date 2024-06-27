import ast
import logging
from utils import all_exists
import traceback
import sys
from typing import List
import subprocess


def remove_parent_dir(formatted_exception_ls: List):
    res = []
    for line in formatted_exception_ls:
        start = line.find('"')
        end = line.rfind('"')
        if (start == -1) or (end == -1):
            res.append(line)
            continue
        start += 1
        file_path = line[start:end]
        file_name = file_path.split('/')[-1]
        res.append(line[:start] + file_name + line[end:])
    return res


def my_ast_parse(code_str):
    tree = None
    err_msg = None

    try:
        tree = ast.parse(code_str)
    except Exception as e:
        logging.exception('code parsing failed:\n' + str(e))

        # this gives the complete traceback with my machine path
        # err_msg = traceback.format_exc()

        # instead, i use this and de-identify the path
        exc_type, exc_value, exc_traceback = sys.exc_info()
        formatted_exception_ls = traceback.format_exception(exc_type, exc_value, exc_traceback)
        exception_ls = remove_parent_dir(formatted_exception_ls)

        err_msg = ''.join(exception_ls)

    return tree, err_msg


def get_all_funcs(code_str, return_str=True):
    tree, err_msg = my_ast_parse(code_str)

    if all_exists(err_msg):
        return None, err_msg

    res = []
    for e in ast.walk(tree):
        if isinstance(e, ast.FunctionDef):
            res.append([e.name, ast.unparse(e) if return_str else e])

    return res, err_msg


def find_func(code_str, func_name, return_str=True):

    all_func_ls, err_msg = get_all_funcs(code_str, return_str=return_str)

    if all_exists(err_msg):
        return None, err_msg, None

    target_func = [func for name, func in all_func_ls if name == func_name]
    if len(target_func) == 0:
        target_func = None
    else:
        target_func = target_func[0]

    return target_func, err_msg, all_func_ls


def run_code(code_str, timeout=10):
    try:
        result = subprocess.run(['python', '-c', code_str], capture_output=True, text=True, timeout=timeout)
        # deidentify the wrong time error stack
        if result.returncode != 0:
            deidented_lines = remove_parent_dir(result.stderr.split('\n'))
            result.stderr = '\n'.join(deidented_lines)
    # handle timeout when running the code
    except subprocess.TimeoutExpired:
        dummy_res = subprocess.CompletedProcess(
            args=['python', '-c', code_str],
            returncode=1,
            stdout='',
            stderr='TimeoutExpired: Command timed out'
        )
        result = dummy_res
    return result