import os
from glob import glob
import multiprocessing
import logging

import numpy as np


def many_to_one(params):
    folder_path, save_path = params[0], params[1]
    file_lst = os.listdir(folder_path)
    final_data_dict = {}
    for i, f in enumerate(file_lst):
        data_dict = np.load(os.path.join(folder_path, f), allow_pickle=True).item()
        if i == 0:
            for k, v in data_dict.items():
                if "obj" not in k:
                    final_data_dict[k] = [v]
        else:
            for k, v in data_dict.items():
                if "obj" not in k:
                    final_data_dict[k].append(v)
    for k in final_data_dict.keys():
        if k != "scene_path":
            final_data_dict[k] = np.stack(final_data_dict[k], axis=0)
        else:
            final_data_dict[k] = final_data_dict[k][0]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, final_data_dict)
    return


def task_collect(configs):
    succ_path_lst = glob(os.path.join(configs.succ_dir, "**/*.npy"), recursive=True)
    succ_folder_lst = list(set([os.path.dirname(p) for p in succ_path_lst]))
    logging.info(
        f"Get {len(succ_path_lst)} success data and {len(succ_folder_lst)} folder."
    )
    iter_param_lst = [
        (f, f.replace(configs.succ_dir, configs.collect_dir) + ".npy")
        for f in succ_folder_lst
    ]
    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(many_to_one, iter_param_lst)
        results = list(result_iter)
    return
