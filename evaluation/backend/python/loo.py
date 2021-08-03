import math
from collections import OrderedDict
import numpy as np
import csv

from utils.stats import Statistics
from .. import LOO_METRICS
# from evaluation.backend import LOO_METRICS


def compute_loo_metrics_py(pred, target, ks, preds_out):
    score_cumulator = OrderedDict()
    for metric in LOO_METRICS:
        score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in ks}
    
    max_k = max(ks)
    with open(preds_out, 'w') as preds_file:
        writer = csv.writer(preds_file, delimiter=',', lineterminator='\n', )
        for idx, u in enumerate(target):
            pred_u = pred[idx]
            target_u = target[u][0]

            loc_target = np.where(pred_u == target_u)[0][0] + 1
            writer.writerow([u, target_u, loc_target])
            hit_at_k = np.where(pred_u == target_u)[0][0] + 1 if target_u in pred_u else max_k + 1

            for k in ks:
                hr_k = 1 if hit_at_k <= k else 0
                ndcg_k = 1 / math.log(hit_at_k + 1, 2) if hit_at_k <= k else 0
                rr_k = (1 / hit_at_k) if hit_at_k <= k else 0

                score_cumulator['HR'][k].update(hr_k)
                score_cumulator['NDCG'][k].update(ndcg_k)
                score_cumulator['MRR'][k].update(rr_k)
    return score_cumulator