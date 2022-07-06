import argparse
import multiprocessing
import time
import os
import h5py
import pickle
import glob
from database import HistoDatabase
from tqdm import tqdm
import faiss
import numpy as np
import pandas as pd
import math

from multiprocessing import Manager, Process
from collections import Counter

# Slides which are in poor quality
IGNORE_SLIDES = ['TCGA-C5-A8YT-01Z-00-DX1.5609D977-4B7E-4B49-A3FB-50434D6E49F9', 
                 'TCGA-06-1086-01Z-00-DX2.e1961f1f-a823-4775-acf7-04a46f05e15e', 
                 'C3N-02678-21']

def eval(results, topk):
    HIT = []
    MV = []
    AP = [] # TODO
    label_set = sorted(list(set([r['label_query'] for r in results.values()])))
    label2id = {label: i for i, label in enumerate(label_set)}
    if args.site == 'organ':
        target_label = 5 # index_meta['site']
    else:
        target_label = 4 # index_meta['diagnosis']
    for query, ans in results.items():
        query_label = ans['label_query']
        ans_labels = Counter([ans['results'][i][target_label] for i in range(topk)])
        ans_dists = np.array([ans['results'][i][1] for i in range(topk)])
        
        # HIT
        HIT.append(1 if query_label in list(ans_labels.keys()) else 0)

        # MV
        max_cnt = max(list(ans_labels.values()))
        label_with_max_cnt = [k for k, v in ans_labels.items() if v == max_cnt]
        if max_cnt > topk // 2 and query_label in label_with_max_cnt:
            MV.append(1)
        else:
            MV.append(0)
    hit_acc = sum(HIT) / len(HIT)
    MV_acc = sum(MV) / len(HIT)
    print(f'Given results has hit acc: {hit_acc}, MV acc: {MV_acc}')
    return hit_acc, MV_acc


def run(pid, latent_path_list_sub, latent_path_list_all, slide_topk, t_total, results):
    pbar = tqdm(latent_path_list_sub)
    t_total_local = 0
    results_local = {}
    for cnt, latent_path in enumerate(pbar):
        resolution = latent_path.split("/")[-3]
        diagnosis = latent_path.split("/")[-4]
        anatomic_site = latent_path.split("/")[-5]
        slide_id = os.path.basename(latent_path).replace(".h5", "")
        if slide_id in IGNORE_SLIDES:
            continue
        patient_id = slide_id.split("-")[2]

        slide_path = os.path.join(args.slide_path, anatomic_site, diagnosis, 
                                resolution, slide_id + ".svs")
        densefeat_path = latent_path.replace("vqvae", "densenet").replace(".h5", ".pkl")
        with open(densefeat_path, 'rb') as handle:
            densefeat = pickle.load(handle)

        t_start = time.time()
        tmp_res = []
        densefeat_byte = np.array(
            [[int(densefeat[patch_idx][j:j+8], 2) for j in range(0, 1024, 8)] for patch_idx in range(len(densefeat))], dtype=np.uint8
        )
        topk = len(densefeat)
        d = 1024
        # build Hamming distance index for search
        index = faiss.IndexBinaryFlat(d)
        index.add(densefeat_byte)
        res_tmp = []

        pbar2 = tqdm(latent_path_list_all)
        build_time = 0
        search_time = 0
        for latent_j in latent_path_list_all:
            slide_id_j = os.path.basename(latent_j).replace(".h5", "")
            if slide_id_j in IGNORE_SLIDES:
                continue
            patient_id_j = slide_id_j.split("-")[2]
            if patient_id_j == patient_id:
                continue
            densefeat_path_j = latent_j.replace("vqvae", "densenet").replace(".h5", ".pkl")
            with h5py.File(latent_j, 'r') as hf:
                feat_j = hf['features'][:]
            with open(densefeat_path_j, 'rb') as handle:
                densefeat_j = pickle.load(handle)

            st = time.time()
            queries_byte = np.array(
                [[int(densefeat_j[patch_idx][j:j+8], 2) for j in range(0, 1024, 8)] for patch_idx in range(len(densefeat_j))], dtype=np.uint8
            )
            build_time += time.time() - st
            st = time.time()
            D, I = index.search(queries_byte, topk)
            min_dist = np.min(D, axis=1)
            median_of_min = np.median(min_dist)

            slide_info = db.query_slide_info(feat_j[0], densefeat_j[0])
            res_tmp.append([slide_id_j, median_of_min] + slide_info)
            res_tmp = sorted(res_tmp, key=lambda x: x[1])
            if len(res_tmp) > slide_topk:
                res_tmp = res_tmp[:-1]
            search_time += time.time() - st
            # pbar2.set_description_str(f'build time: {build_time}, search time: {search_time}')
        t_elapse = time.time() - t_start
        with open(os.path.join(speed_record_path, "speed_log.txt"), 'a') as fw:
            fw.write(slide_id + ", " + str(t_elapse) + "\n")
        t_total_local += t_elapse
        pbar.set_description_str(f'PID: {pid}, time: {t_elapse:.2f}')

        key = slide_id
        results_local[key] = {'results': None, 'label_query': None}
        results_local[key]['results'] = tmp_res # topk slide result with median-of-min hamming dist
        if args.site == 'organ':
            results_local[key]['label_query'] = anatomic_site
        else:
            results_local[key]['label_query'] = diagnosis

        # recover meta
        db.leave_one_patient_fast_recov(patient_id)
    results.update(results_local)
    t_total.value += t_total_local


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Search for WSI query in the database")
    parser.add_argument("--slide_path", type=str, default="./DATA/WSI", 
                        help="The path to all slides")
    parser.add_argument("--latent_path", type=str, default="./DATA/LATENT", 
                        help="The path to all mosaic latent code and text features")
    parser.add_argument("--site", type=str, required=True, 
                        help="The site where the database is built")
    parser.add_argument("--db_index_path", type=str, required=True, 
                        help="Path to the veb tree that stores all indices")
    parser.add_argument("--index_meta_path", type=str, required=True, 
                        help="Path to the meta data of each index")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of process to run the search")
    parser.add_argument("--topk", type=int, default=3,
                        help="topk slide search result")
    parser.add_argument("--codebook_semantic", type=str, default="./checkpoints/codebook_semantic.pt", 
                        help="Path to the semantic codebook from vq-vae")
    
    # >>>>>>>>>>>>>> Update for csv >>>>>>>>>>>>>>>>>
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--fold_num", type=int, default=24)
    args = parser.parse_args()
    # >>>>>>>>>>>>>> Update for csv >>>>>>>>>>>>>>>>>

    if args.site == 'organ':
        save_path = os.path.join("QUERY_RESULTS", args.site)
        latent_all = os.path.join(args.latent_path, "*", "*", "*", "vqvae", "*")
        speed_record_path = os.path.join("QUERY_SPEED", args.site)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(speed_record_path):
            os.makedirs(speed_record_path)
    else:
        save_path = os.path.join("QUERY_RESULTS", args.site)
        latent_all = os.path.join(args.latent_path, args.site, 
                                  "*", "*", "vqvae", "*")
        speed_record_path = os.path.join("QUERY_SPEED", args.site)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(speed_record_path):
            os.makedirs(speed_record_path)

    db = HistoDatabase(database_index_path=args.db_index_path, 
                       index_meta_path=args.index_meta_path, 
                       codebook_semantic=args.codebook_semantic)

    # >>>>>>>>>>>>>> Update for csv >>>>>>>>>>>>>>>>>
    mosaic_all = pd.read_csv(r'/mnt/ceph_sz/private/scusenyang/data_tencent/MedIA_search/FISH/ys/site/sen_extra_search.csv')

    # ---- split folds ----
    total_num = len(mosaic_all)
    fold_size = math.ceil(total_num / args.fold_num)

    mosaic_all = pd.DataFrame(data=mosaic_all.values[args.index * fold_size : args.index * fold_size + fold_size], columns=mosaic_all.columns)

    # Get all latent path
    latent_path_all = mosaic_all['id']
    # latent_path_all = glob.glob(latent_all)
    # >>>>>>>>>>>>>> Update for csv >>>>>>>>>>>>>>>>>

    t_total = multiprocessing.Value('f', 0)
    manager = Manager()
    results = manager.dict()

    # Split to each workers
    list_latent_path_sub = [[] for _ in range(args.num_workers)]
    for i, latent_path in enumerate(latent_path_all):
        list_latent_path_sub[i % args.num_workers].append(latent_path)

    ps = []
    for pid, latent_path_sub in enumerate(list_latent_path_sub):
        p = Process(target=run, args=(pid, latent_path_sub, latent_path_all, args.topk, t_total, results))
        p.start()
        ps.append(p)

    print('>>>>>>>>>> PROCESS >>>>>>>>>>')
    [p.join() for p in ps]
    print('>>>>>>>>>> DONE >>>>>>>>>>')
    t_acc = t_total.value
    total_res = results

    print("Total search takes: ", t_acc)
    # >>>>>>>>>>>>>> Update for csv >>>>>>>>>>>>>>>>>
    with open(os.path.join(save_path, f"results_{args.index}.pkl"), 'wb') as handle:
        pickle.dump(total_res, handle)
    # >>>>>>>>>>>>>> Update for csv >>>>>>>>>>>>>>>>>
