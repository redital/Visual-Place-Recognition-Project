import numpy as np

def read_file_preds(preds_txt_file):
    with open(preds_txt_file) as file:
        lines = file.read().splitlines()
    query_path = lines[1]
    preds_paths = lines[4:lines.index('', 4)]
    positives_paths = lines[26:] # extracts also the paths of the retrieved images that are correct
    
    return query_path, preds_paths, positives_paths

def get_utm_from_path(path):
    return np.array([path.split("@")[1], path.split("@")[2]]).astype(np.float32)

def compute_distance(point_A, point_B):
    return ((point_A - point_B) ** 2).sum() ** 0.5

def get_list_distances_from_preds(preds_txt_file):
    query_path, preds_paths, _ = read_file_preds(preds_txt_file)
    query_utm = get_utm_from_path(query_path)
    list_preds_utm = [get_utm_from_path(pred_path) for pred_path in preds_paths]
    distances = [compute_distance(query_utm, pred_utm) for pred_utm in list_preds_utm]
    return distances  # Distances are in meters
