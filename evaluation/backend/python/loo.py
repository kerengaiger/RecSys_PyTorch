import math
from collections import OrderedDict
import numpy as np
import pandas as pd
import csv

from utils.stats import Statistics
from .. import LOO_METRICS
# from evaluation.backend import LOO_METRICS


def compute_loo_metrics_py(pred, target, ks, usermap_file, itemmap_file):
    score_cumulator = OrderedDict()
    for metric in LOO_METRICS:
        score_cumulator[metric] = {k: Statistics('%s@%d' % (metric, k)) for k in ks}

    usermap = pd.read_csv(usermap_file, names=['usr_old', 'usr_new'])
    itemmap = pd.read_csv(itemmap_file, names=['itm_old', 'itm_new'])

    max_k = max(ks)
    hits_at_k = []
    users = []
    items = []
    for idx, u in enumerate(target):

        pred_u = pred[idx]
        target_u = target[u][0]

        hit_at_k = np.where(pred_u == target_u)[0][0] + 1
        hits_at_k.append(hit_at_k)
        users.append(u)
        items.append(target_u)

        for k in ks:
            hr_k = 1 if hit_at_k <= k else 0
            ndcg_k = 1 / math.log(hit_at_k + 1, 2) if hit_at_k <= k else 0
            rr_k = (1 / hit_at_k) if hit_at_k <= k else 0

            score_cumulator['HR'][k].update(hr_k)
            score_cumulator['NDCG'][k].update(ndcg_k)
            score_cumulator['MRR'][k].update(rr_k)
    preds_df = pd.DataFrame({'user': [usermap.loc[usermap['usr_new'] == u, 'usr_old'].values[0] for u in users],
                            'item': [itemmap.loc[itemmap['itm_new'] == i, 'itm_old'].values[0] for i in items],
                            'pred_loc': hits_at_k})

    return score_cumulator, preds_df