import os
import multiprocessing
import logging
from glob import glob
import traceback

import numpy as np

from .control_eval_func import *


def safe_eval_one(params):
    input_npy_path, configs = params[0], params[1]
    try:
        if configs.task.method == "op":
            eval_func_name = f"{configs.setting}DummyArmOpEval"
        elif configs.task.method == "ours":
            eval_func_name = f"{configs.setting}DummyArmOursEval"
        elif configs.task.method == "bs1":
            eval_func_name = f"{configs.setting}DummyArmBS1Eval"
        elif configs.task.method == "bs2":
            eval_func_name = f"{configs.setting}DummyArmBS2Eval"
        elif configs.task.method == "bs3":
            eval_func_name = f"{configs.setting}DummyArmBS3Eval"
        else:
            raise NotImplementedError()
        eval(eval_func_name)(input_npy_path, configs).run()
        return
    except Exception:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
        return


def task_control_eval(configs):
    input_dir = configs[configs.task.input_data]
    input_path_lst = glob(os.path.join(input_dir, "**/*.npy"), recursive=True)
    init_num = len(input_path_lst)

    if configs.skip:
        eval_path_lst = glob(os.path.join(configs.eval_dir, "**/*.npy"), recursive=True)
        eval_path_lst = [p.replace(configs.eval_dir, input_dir) for p in eval_path_lst]
        input_path_lst = list(set(input_path_lst).difference(set(eval_path_lst)))
    skip_num = init_num - len(input_path_lst)
    input_path_lst = sorted(input_path_lst)
    if configs.task.max_num > 0:
        input_path_lst = np.random.permutation(input_path_lst)[: configs.task.max_num]
        input_path_lst = sorted(input_path_lst)

    logging.info(f"Find {init_num} grasp data in {input_dir}, skip {skip_num}, and use {len(input_path_lst)}.")

    if len(input_path_lst) == 0:
        return

    iterable_params = zip(input_path_lst, [configs] * len(input_path_lst))
    if configs.task.debug_viewer or configs.task.debug_render:
        for i, ip in enumerate(iterable_params):
            # to run a specific case
            if i >= 0:
                print(f"grasp sample id: {i}")
                safe_eval_one(ip)
    else:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(safe_eval_one, iterable_params)
            results = list(result_iter)

    logging.info("Finish control evaluation")

    return
