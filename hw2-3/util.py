"""
My homework helper module
"""

import os
from typing import *

script_dir = os.path.realpath(__file__)

def run_process(run_list: list[str],
                resolve: Callable = lambda: 0,
                reject: Callable[[str], None] = lambda: None):
    """
    General process triggering function
    :param run_list: command in list format to be run, \
                     e.g. `pip install meow` will become `["pip", "install", "meow]`
    """
    import subprocess
    res = subprocess.run(run_list,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    if res.returncode == 0:
        resolve()
    else:
        reject(res.stderr)

def download_file(url: str, name: str):
    """
    Download file w.r.t. given url
    """
    run_process(["wget", url, "-O", name],
                lambda: print(f"Download and save {name} successfully", end="\n\n"),
                lambda err: print(f"Failed to download from url {url}: {err}", end="\n\n"))

def install_dependency(dep: List[str]):
    """
    Install dependencies for this python scripts
    """
    run_process(["pip", "install", *dep],
                lambda: print(f"Install dependenc{'y' if len(dep) == 1 else 'ies'}: [", *dep, "] successfully", end="\n\n"),
                lambda err: print("Failed to install dependencies:", *dep, err, end="\n\n"))
    
def print_hw_result(mark: str, title: str, *lines: List[str]):
    """
    Print homework result and write corresponding file
    """
    def __print_to(file=None):
        print(f"[{mark}] {title}",
              *[f'\t{l}' for l in lines], sep="\n", end="\n\n", file=file)
        
    __print_to()
    with open(f"{os.path.dirname(script_dir)}/hw{mark}-output.txt", 'w') as f:
        __print_to(f)

def flatten(container: Union[List, set]):
    """
    Flatten a list/set

    Return: same class containing flatten elements
    """
    res = container.__class__()
    for c in container:
        if type(c) not in [list, set]:
            if isinstance(res, list):
                res.append(c)
            elif isinstance(res, set):
                res.add(c)
            else:
                print(f"Error: container type {type(res)} not support")
                exit(1)
        else:
            res.extend(flatten(c))
    return res

def json_stringify(json: Union[dict, list], indent = 0) -> list[str]:
    """
    Convert json-structured container of type `Union[dict, list]`
    to a list of string with auto indention.
    
    :param json: if is a `list`, then it containes a list of `dict`
    """
    res = []
    def __json_stringify(json: Union[dict, list], indent = 0):
        if type(json) == list:
            for j in json:
                __json_stringify(j, indent)
        else:
            key_lim = max(len(str(k)) for k in json.keys()) + 1
            for k, v in json.items():
                if type(v) in (list, dict): # run recursively
                    res.append("\t" * indent + f"{k}")
                    __json_stringify(v, indent + 1)
                else:
                    res.append("\t" * indent + f"{k}{' ' * (key_lim - len(str(k)))}: {v}")
    
    __json_stringify(json, indent)
    return res

def extract_dict(single_element_dict: dict) -> tuple:
    """
    Un-wrap a dict where there is only one element inside it
    """
    if len(single_element_dict) != 1:
        raise ValueError("Can only extract dict with only one entry")
    return [*single_element_dict.items()][0]
    
    