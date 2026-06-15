import os
import sys
import argparse
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import numpy as np
import joblib
from util import read_file_preds

sys.path.append(str(Path(__file__).parent.joinpath("image-matching-models")))

from matching import get_matcher, available_models
from matching.utils import get_default_device

def logistic_regressor_predict(num_inliers, model_path):
    num_inliers = np.asarray(num_inliers).reshape(-1,1)
    
    clf = joblib.load(model_path)
    return clf.predict(num_inliers)

def threshold_predict(num_inliers, threshold):
    return num_inliers > threshold

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--out-dir", type=str, default=None, help="output directory of image matching results")
    parser.add_argument("--matcher", type=str, default="sift-lg", choices=available_models, help="choose your matcher")
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--im-size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to match")
    parser.add_argument("--start-query", type=int, default=-1, help="query to start from")
    parser.add_argument("--num-queries", type=int, default=-1, help="number of queries")
    parser.add_argument("--adaptive", type=bool, default=False, help="re-ranking adattivo o no (lasciare vuoto per no)")
    parser.add_argument("--threshold", type=int, default=-1, help="soglia")
    parser.add_argument("--model_type", type=str, default="regressor",  choices=["threshold", "regressor"])
    parser.add_argument("--model_path", help="Percorso del modello trainato")
    
    return parser.parse_args()

def main(args):
    device = args.device
    matcher_name = args.matcher
    img_size = args.im_size
    num_preds = args.num_preds
    matcher = get_matcher(matcher_name, device=device)
    preds_folder = args.preds_dir
    start_query = args.start_query
    num_queries = args.num_queries
    threshold = args.threshold
    adaptive = args.adaptive
    model_type = args.model_type
    model_path = Path(args.model_path)
    if model_type == "regressor" and not model_path:
        raise Exception("if you want to use a regressor you must provide the model path")
    elif model_type == "threshold" and not threshold:
        raise Exception("if you want to use a hard threshold you must provide the threshold")

    output_folder = Path(preds_folder + f"_{matcher_name}") if args.out_dir is None else Path(args.out_dir)
    output_folder.mkdir(exist_ok=True)
    
    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    start_query = start_query if start_query >= 0 else 0
    num_queries = num_queries if num_queries >= 0 else len(txt_files)

    controlli_effettivi = 0

    for txt_file in tqdm(txt_files[start_query : start_query + num_queries]):
        q_num = Path(txt_file).stem
        out_file = output_folder.joinpath(f"{q_num}.torch")
        if out_file.exists():
            continue

        results = []
        q_path, pred_paths, _ = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=img_size)
        for pred_path in pred_paths[:num_preds]:
            img1 = matcher.load_image(pred_path, resize=img_size)
            result = matcher(deepcopy(img0), img1)
            controlli_effettivi += 1
            if adaptive:
                if model_type == "regressor":
                    if logistic_regressor_predict(result['num_inliers'], model_path):
                        break
                if model_type == "threshold":
                    if threshold_predict(result['num_inliers'], threshold):
                        break
            result["all_desc0"] = result["all_desc1"] = None
            results.append(result)
        torch.save(results, out_file)
    print("controlli attesi: ", num_queries*num_preds)
    print("controlli effettivi: ", controlli_effettivi)
    print("saltati {} re-ranking".format(num_queries*num_preds - controlli_effettivi))

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
