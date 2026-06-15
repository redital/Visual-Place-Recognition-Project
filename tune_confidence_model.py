import os
import torch
import numpy as np
import argparse
from glob import glob
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
from tqdm import tqdm
import joblib
from util import read_file_preds

def build_dataset(preds_dir: str, inliers_dir: str):
    txt_files = glob(os.path.join(preds_dir, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    X, y = [], []

    for txt_file in tqdm(txt_files, leave=False):
        q_num = Path(txt_file).stem
        torch_file = Path(inliers_dir) / f"{q_num}.torch"

        if not torch_file.exists():
            continue

        _, pred_paths, pos_paths = read_file_preds(txt_file)
        results = torch.load(torch_file, weights_only=False)

        # feature: una riga per predizione (scalare intero num_inliers) 
        # label: 1 se quella predizione è vera
        for i, r in enumerate(results):
            X.append(r["num_inliers"])
            label = int(pred_paths[i] in pos_paths)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def tune(X_train, y_train, X_val, y_val, model_path: Path):
    X_train = np.asarray(X_train).reshape(-1, 1)
    X_val   = np.asarray(X_val).reshape(-1, 1)

    X_all = np.vstack((X_train, X_val))
    y_all = np.concatenate((y_train, y_val))

    # -1 = train, 0 = val 
    fold = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_val))
    ])
    pds = PredefinedSplit(test_fold=fold)

    param_grid = {
        "C":            [0.01, 0.1, 1.0, 10.0, 100.0],
        "class_weight": ["balanced", None],
        "max_iter":     [100, 1000, 10000],
    }

    grid = RandomizedSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=pds,
        scoring="roc_auc",
        n_jobs=-1,
            )
    grid.fit(X_all, y_all)

    print(f"  Migliori parametri : {grid.best_params_}")
    print(f"  Miglior AUC su Val : {grid.best_score_:.4f}")

    joblib.dump(grid.best_estimator_, model_path)
    return grid.best_estimator_, grid.best_params_

def main(args):
    train_base = Path(args.train_base_dir)
    val_base   = Path(args.val_base_dir)
    model_dir  = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train_all, y_train_all = [], []
    X_val_all,   y_val_all   = [], []

    for vpr_dir in sorted(train_base.iterdir()):
        if not vpr_dir.is_dir():
            continue
        vpr_method = vpr_dir.name

        matcher_dirs = sorted(
            d for d in vpr_dir.iterdir()
            if d.is_dir() and d.name.startswith("preds_")
        )

        for matcher_dir in matcher_dirs:
            matcher = matcher_dir.name.replace("preds_", "")
            combo   = f"{vpr_method}__{matcher}"

            # controlla che il val set esista 
            val_vpr_dir = val_base / vpr_method

            preds_val   = val_vpr_dir / "preds"
            matched_val = val_vpr_dir / matcher_dir.name
            if not preds_val.exists() or not matched_val.exists():
                print(f"\n=== {combo} === SKIP (val set mancante)")
                continue

            # costruisce i dataset per questa combo 
            print(f"\n=== {combo} ===")
            X_tr, y_tr = build_dataset(str(vpr_dir / "preds"), str(matcher_dir))
            X_vl, y_vl = build_dataset(str(preds_val),           str(matched_val))
            assert len(X_tr) == len(y_tr), f"{combo} TRAIN: X={len(X_tr)} y={len(y_tr)}"
            assert len(X_vl) == len(y_vl), f"{combo} VAL:   X={len(X_vl)} y={len(y_vl)}"

            print(f"  Train: {len(X_tr)} | Val: {len(X_vl)}")

            X_train_all.append(X_tr)
            y_train_all.append(y_tr)
            X_val_all.append(X_vl)
            y_val_all.append(y_vl)

    if not X_train_all:
        raise RuntimeError("Nessun dataset trovato. Controlla i percorsi.")

    # dataset unico: concatena lungo l'asse delle query 
    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_val   = np.concatenate(X_val_all,   axis=0)
    y_val   = np.concatenate(y_val_all,   axis=0)

    print(f"\n{'='*50}")
    print(f"Dataset globale — Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"{'='*50}\n")

    model_path = model_dir / f"{train_base.name}.pkl"
    tune(X_train, y_train, X_val, y_val, model_path)
    print(f"\nModello salvato in: {model_path}")
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_base_dir", required=True)
    parser.add_argument("--val_base_dir",   required=True)
    parser.add_argument("--model_dir",      required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
