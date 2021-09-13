import math
import time
import torch
import pathlib
import numpy as np
import os
import pickle
from collections import OrderedDict

from utils.Tools import RunningAverage as AVG
from utils.backend import predict_topk, compute_holdout, CPP_AVAILABLE

class Evaluator:
    def __init__(self, eval_pos, eval_target, item_popularity, top_k):
        self.top_k = top_k if isinstance(top_k, list) else [top_k]
        self.max_k = max(self.top_k)
        self.eval_pos = eval_pos
        self.eval_target = eval_target
        self.item_popularity = item_popularity
        self.num_users, self.num_items = self.eval_pos.shape
        self.item_self_information = self.compute_item_self_info(item_popularity)

    def hit_ratio_k(self, model, test_batch_size, data_name, data_dir, save_dir):
        model.eval()

        model.before_evaluate()

        eval_users = np.array(list(self.eval_target.keys()))

        pred_matrix = model.predict(eval_users, self.eval_pos, test_batch_size)

        topk = predict_topk(pred_matrix.astype(np.float32), max(self.top_k))

        data_path = os.path.join(data_dir, data_name, data_name + '.data')
        data = pickle.load(open(data_path, 'rb'))
        item2id = data['item_id_dict']
        id2item = {v: k for k, v in item2id.items()}
        user2id = data['user_id_dict']
        id2user = {v: k for k, v in user2id.items()}
        res = {}
        for k in self.top_k:
            with open(pathlib.Path(save_dir, f'hr_{data_name}_{k}.csv'), 'w') as hr_file:
                hits = 0
                for usr_id in self.eval_target.keys():
                    if self.eval_target[usr_id] in topk[usr_id, :k]:
                        hr_file.write(f'{str(id2user[usr_id])}, {str(id2item[self.eval_target[usr_id][0]])}, 1')
                        hr_file.write('\n')
                        hits += 1
                    else:
                        hr_file.write(f'{str(id2user[usr_id])}, {str(id2item[self.eval_target[usr_id][0]])}, 0')
                        hr_file.write('\n')
                res[k] = hits / len(self.eval_target.keys())
        return res

    def mrr_k(self, model, test_batch_size, data_name, data_dir, save_dir):
        model.eval()

        model.before_evaluate()

        eval_users = np.array(list(self.eval_target.keys()))

        pred_matrix = model.predict(eval_users, self.eval_pos, test_batch_size)

        topk = predict_topk(pred_matrix.astype(np.float32), pred_matrix.shape[1])

        data_path = os.path.join(data_dir, data_name, data_name + '.data')
        data = pickle.load(open(data_path, 'rb'))
        item2id = data['item_id_dict']
        id2item = {v: k for k, v in item2id.items()}
        user2id = data['user_id_dict']
        id2user = {v: k for k, v in user2id.items()}

        res = {}
        for k in self.top_k:
            with open(pathlib.Path(save_dir, f'rr_{data_name}_{k}.csv'), 'w') as rr_file:
                cum_rr = 0
                for usr_id in self.eval_target.keys():
                    loc = np.where(topk[usr_id, :] == self.eval_target[usr_id])[0][0] + 1
                    if self.eval_target[usr_id] in topk[usr_id, :k]:
                        cur_rank = 1 / loc
                        rr_file.write(f'{str(id2user[usr_id])}, {str(id2item[self.eval_target[usr_id][0]])}, '
                                      f'{cur_rank}, {loc}')
                        rr_file.write('\n')
                        cum_rr += cur_rank
                    else:
                        rr_file.write(f'{str(id2user[usr_id])}, {str(id2item[self.eval_target[usr_id][0]])}, 0, {loc}')
                        rr_file.write('\n')
                res[k] = cum_rr / len(self.eval_target.keys())
        return res

    def evaluate(self, model, dataset, test_batch_size, mean=True, return_topk=False):
        model.eval()

        model.before_evaluate()

        eval_users = np.array(list(self.eval_target.keys()))

        pred_matrix = model.predict(eval_users, self.eval_pos, test_batch_size)

        topk = predict_topk(pred_matrix.astype(np.float32), max(self.top_k))
        
        # Precision, Recall, NDCG @ k
        scores = self.prec_recall_ndcg(topk, self.eval_target)
        score_dict = OrderedDict()
        for metric in scores:
            score_by_ks = scores[metric]
            for k in score_by_ks:
                if mean:
                    score_dict['%s@%d' % (metric, k)] = score_by_ks[k].mean
                else:
                    score_dict['%s@%d' % (metric, k)] = score_by_ks[k].history
        
        # Novelty @ k
        novelty_dict = self.novelty(topk)
        for k, v in novelty_dict.items():
            score_dict[k] = v

        # Gini diversity
        score_dict['Gini-D'] = self.gini_diversity(topk)

        if return_topk:
            return score_dict, topk
        else:
            return score_dict

    def predict_topk(self, scores, k):
        # top_k item index (not sorted)
        relevant_items_partition = (-scores).argpartition(k, 1)[:, 0:k]
        
        # top_k item score (not sorted)
        relevant_items_partition_original_value = np.take_along_axis(scores, relevant_items_partition, 1)
        
        # top_k item sorted index for partition
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, 1)
        
        # sort top_k index
        topk = np.take_along_axis(relevant_items_partition, relevant_items_partition_sorting, 1)

        return topk

    def prec_recall_ndcg(self, topk, target):
        if CPP_AVAILABLE:
            results = compute_holdout(topk.astype(np.int32), target, 3, np.array(self.top_k, dtype=np.int32))

            prec = {k: AVG() for k in self.top_k}
            recall = {k: AVG() for k in self.top_k}
            ndcg = {k: AVG() for k in self.top_k}
            scores = {
                'Prec': prec,
                'Recall': recall,
                'NDCG': ndcg
            }

            for idx, u in enumerate(target):
                user_results = results[idx].tolist()
                for i, metric in enumerate(['Prec', 'Recall', 'NDCG']):
                    for j, k in enumerate(self.top_k):
                        scores[metric][k].update(user_results[i * len(self.top_k) + j])
        else:
            scores = compute_holdout(topk.astype(np.int32), target, self.top_k)
        return scores

    def novelty(self, topk):
        topk_info = np.take(self.item_self_information, topk)
        top_k_array = np.array(self.top_k)
        topk_info_sum = np.cumsum(topk_info, 1)[:, top_k_array - 1]
        novelty_all_users = topk_info_sum / np.atleast_2d(top_k_array)
        novelty = np.mean(novelty_all_users, axis=0)

        novelty_dict = {'Nov@%d' % self.top_k[i]: novelty[i] for i in range(len(self.top_k))}

        return novelty_dict

    def gini_diversity(self, topk):
        num_items = self.eval_pos.shape[1]
        item_recommend_counter = np.zeros(num_items, dtype=np.int)
        
        rec_item, rec_count = np.unique(topk, return_counts=True)
        item_recommend_counter[rec_item] += rec_count

        item_recommend_counter_mask = np.ones_like(item_recommend_counter, dtype = np.bool)
        item_recommend_counter_mask[item_recommend_counter == 0] = False
        item_recommend_counter = item_recommend_counter[item_recommend_counter_mask]
        num_eff_items = len(item_recommend_counter)

        item_recommend_counter_sorted = np.sort(item_recommend_counter)       # values must be sorted
        index = np.arange(1, num_eff_items+1)                                 # index per array element

        gini_diversity = 2 * np.sum((num_eff_items + 1 - index) / (num_eff_items + 1) * item_recommend_counter_sorted / np.sum(item_recommend_counter_sorted))
        return gini_diversity

    def compute_item_self_info(self, item_popularity):
        self_info = np.zeros(len(item_popularity))
        # total = 0
        for i in item_popularity:
            self_info[i] = item_popularity[i] / self.num_users
            # total += item_popularity[i]
        # self_info /= total
        self_info = -np.log2(self_info)
        return self_info