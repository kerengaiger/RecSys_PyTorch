import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import pickle
import numpy as np
import argparse
import torch
import pathlib

import models
from utils.Params import Params
from utils.Dataset import Dataset
from utils.Logger import Logger
from utils.Evaluator import Evaluator
from utils.Trainer import Trainer
from ax.service.managed_loop import optimize


def train_with_conf(conf):
    model_conf = Params(os.path.join(conf['conf_dir'], f'{conf["model"].lower()}_{conf["data_name"]}.json'))
    for k in conf.keys():
        model_conf.update_dict(k, conf[k])

    np.random.seed(conf['seed'])
    torch.random.manual_seed(conf['seed'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

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

    # logger.info(model_conf)
    # logger.info(dataset)

    trainer = Trainer(
        dataset=dataset,
        model=model,
        evaluator=evaluator,
        logger=logger,
        conf=model_conf
    )
    best_score, best_epoch = trainer.train()
    # calculate hit ratio and mrr
    hr_k = evaluator.hit_ratio_k(model, model_conf.test_batch_size, model_conf.data_name, conf['save_dir'])
    mrr_k = evaluator.mrr_k(model, model_conf.test_batch_size, model_conf.data_name, conf['save_dir'])
    print(f'hr_{evaluator.max_k}:{hr_k}')
    print(f'mrr_{evaluator.max_k}:{mrr_k}')
    if not conf['early_stop']:
        torch.save(model, pathlib.Path(conf['save_dir'], model_conf.data_name + '_bpr.pt'))
    return {'validation loss': (best_score, 0.0), 'best_epoch': (best_epoch, 0.0)}


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='EASE')
parser.add_argument('--data_name', type=str, default='ml-1m')
parser.add_argument('--tune', action='store_true')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./saves')
parser.add_argument('--conf_dir', type=str, default='./conf')
parser.add_argument('--seed', type=int, default=428)

conf = parser.parse_args()

if conf.tune:
    best_parameters, values, _experiment, _cur_model = optimize(
                parameters=[
                    {"name": "learning_rate", "type": "range", "value_type": "float", "bounds": [5e-3, 1e-3]},
                    {"name": "hidden_dim", "type": "choice", "value_type": "int", "values": [12, 17, 20, 25, 30, 50, 100]},
                    {"name": "batch_size", "type": "choice", "value_type": "int", "values": [1024, 4096, 3200, 6000]},
                    {"name": "data_name", "type": "fixed", "value_type": "str", "value": conf.data_name},
                    {"name": "conf_dir", "type": "fixed", "value_type": "str", "value": conf.conf_dir},
                    {"name": "model", "type": "fixed", "value_type": "str", "value": conf.model},
                    {"name": "seed", "type": "fixed", "value_type": "int", "value": conf.seed},
                    {"name": "data_dir", "type": "fixed", "value_type": "str", "value": conf.data_dir},
                    {"name": "save_dir", "type": "fixed", "value_type": "str", "value": conf.save_dir},
                    {"name": "use_validation", "type": "fixed", "value_type": "bool", "value": True},
                    {"name": "early_stop", "type": "fixed", "value_type": "bool", "value": True},

                ],
                evaluation_function=train_with_conf,
                minimize=True,
                objective_name='validation loss',
                total_trials=5
            )
    print('Final Train')
    best_parameters['best_epoch'] = values[0]['best_epoch']
    print(best_parameters['best_epoch'])
    best_parameters['num_epochs'] = int(best_parameters['best_epoch'])
    best_parameters['use_validation'] = False
    best_parameters['early_stop'] = False
    train_with_conf(best_parameters)


else:
    print('need to implement train')

