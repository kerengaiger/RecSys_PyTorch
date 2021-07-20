import os
import pickle
import numpy as np
import argparse
import torch

import models
from utils.Params import Params
from utils.Dataset import Dataset
from utils.Logger import Logger
from utils.Evaluator import Evaluator
from utils.Trainer import Trainer
from ax.service.managed_loop import optimize


def train_with_conf(conf):
    model_conf = Params(os.path.join(conf['conf_dir'], conf['model'].lower() + '.json'))
    for k in conf.keys():
        model_conf.update_dict(k, conf[k])
    print(model_conf)

    np.random.seed(conf['seed'])
    torch.random.manual_seed(conf['seed'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = Dataset(
        data_dir=conf['data_dir'],
        data_name=model_conf.data_name,
        train_ratio=0.8,
        device=device,
        min_usr_len=model_conf.min_usr_len,
        max_usr_len=model_conf.max_usr_len,
        min_item_cnt=model_conf.min_item_cnt,
        max_item_cnt=model_conf.max_item_cnt,
        fin_min_usr_len=model_conf.fin_min_usr_len,
        pos_thresh=model_conf.pos_thresh,
        use_validation=model_conf.use_validation
    )

    log_dir = os.path.join('saves', conf['model'])
    logger = Logger(log_dir)
    model_conf.save(os.path.join(logger.log_dir, 'config.json'))

    eval_pos, eval_target = dataset.eval_data()
    item_popularity = dataset.item_popularity
    evaluator = Evaluator(eval_pos, eval_target, item_popularity, model_conf.top_k)

    model_base = getattr(models, conf['model'])
    model = model_base(model_conf, dataset.num_users, dataset.num_items, device)

    logger.info(model_conf)
    logger.info(dataset)

    trainer = Trainer(
        dataset=dataset,
        model=model,
        evaluator=evaluator,
        logger=logger,
        conf=model_conf
    )
    best_score, best_epoch = trainer.train()
    return {'best_score': (best_score, 0.0), 'best_epoch': (best_epoch, 0.0)}


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='EASE')
parser.add_argument('--tune', action='store_true')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./saves')
parser.add_argument('--conf_dir', type=str, default='./conf')
parser.add_argument('--seed', type=int, default=428)

conf = parser.parse_args()

if conf.tune:
    best_parameters, values, _experiment, _cur_model = optimize(
                parameters=[
                    {"name": "learning_rate", "type": "range", "value_type": "float", "bounds": [5e-3, 5e-2]},
                    {"name": "hidden_dim", "type": "choice", "value_type": "int", "values": [20, 32, 50, 64, 100]},
                    {"name": "batch_size", "type": "choice", "value_type": "int", "values": [4096, 3200, 1000, 6000]},
                    {"name": "conf_dir", "type": "fixed", "value_type": "str", "value": conf.conf_dir},
                    {"name": "model", "type": "fixed", "value_type": "str", "value": conf.model},
                    {"name": "seed", "type": "fixed", "value_type": "int", "value": conf.seed},
                    {"name": "data_dir", "type": "fixed", "value_type": "str", "value": conf.data_dir},
                    {"name": "save_dir", "type": "fixed", "value_type": "str", "value": conf.save_dir},
                    {"name": "use_validation", "type": "fixed", "value_type": "bool", "values": True},
                    {"name": "early_stop", "type": "fixed", "value_type": "bool", "values": True},

                ],
                evaluation_function=train_with_conf,
                minimize=False,
                objective_name='ndcg_score',
                total_trials=5
            )
    best_parameters['best_epoch'] = values[0]['best_epoch']
    # pickle.dump(best_parameters, open(args.cnfg_out, "wb"))
    best_parameters['num_epochs'] = best_parameters['best_epoch']
    best_parameters['use_validation'] = False
    best_parameters['early_stop'] = False
    train_with_conf(best_parameters)

else:
    print('need to implement train')

