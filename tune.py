import os
import torch

import models
from data.dataset import UIRTDataset
from evaluation.evaluator import Evaluator
from experiment.early_stop import EarlyStop

from loggers import FileLogger, CSVLogger
from utils.general import make_log_dir, set_random_seed
from config import load_config

from ax.service.managed_loop import optimize
import argparse


def train_with_conf(hparams_cnfg):
    config = load_config()
    config = {'dataset': {'data_path': 'datasets/amazon/amazonbeauty_corpus.csv', 'dataname': 'amazonbeauty',
                          'separator': ',', 'binarize_threshold': 4.0, 'implicit': True, 'min_usr_len': 2,
                          'max_usr_len': 1000, 'min_items_cnt': 5, 'max_items_cnt': 50000, 'final_usr_len': 4,
                          'protocol': 'leave_one_out', 'generalization': 'weak', 'holdout_users': 0,
                          'valid_ratio': 0.0, 'test_ratio': 0.0, 'leave_k': 1, 'split_random': False,
                          'use_validation': False},
              'evaluator': {'ks': [20, 10, 5]}, 'early_stop': {'early_stop': 40, 'early_stop_measure': 'HR@20'},
              'experiment': {'debug': False, 'save_dir': 'saves', 'num_epochs': 300, 'batch_size': 128, 'verbose':0,
                             'print_step': 1, 'test_step': 1, 'test_from': 1, 'model_name': 'LightGCN', 'num_exp': 5,
                             'seed': 2020, 'gpu': 1}, 'hparams': {'node_dropout': 0.3905314184725284,
                                                                  'emb_dim': 128, 'num_layers': 5, 'split': False,
                                                                  'num_folds': 100, 'graph_dir': 'graph', 'reg': 0.0001,
                                                                  'use_validation': False}}

    # exp_config = config.experiment
    exp_config = config['experiment']
    # gpu_id = exp_config.gpu
    gpu_id = exp_config['gpu']
    seed = exp_config['seed']
    # seed = exp_config.seed

    # dataset_config = config.dataset
    dataset_config = config['dataset']
    dataset_config['use_validation'] = hparams_cnfg['use_validation']

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_random_seed(seed)

    """ 
        Dataset
    """
    dataset = UIRTDataset(**dataset_config)

    """ 
        Early stop
    """
    # early_stop = EarlyStop(**config.early_stop)
    early_stop = EarlyStop(**config['early_stop'])


    """ 
        Model base class
    """
    # model_name = config.experiment.model_name
    model_name = config['experiment']['model_name']
    model_base = getattr(models, model_name)
    # log_dir = make_log_dir(os.path.join(exp_config.save_dir, model_name))
    log_dir = make_log_dir(os.path.join(exp_config['save_dir'], model_name))
    logger = FileLogger(log_dir)
    csv_logger = CSVLogger(log_dir)
    config.hparams = hparams_cnfg

    # Save log & dataset config.
    logger.info(config)
    logger.info(dataset)

    valid_input, valid_target = dataset.valid_input, dataset.valid_target
    # evaluator = Evaluator(valid_input, valid_target, dataset_config.dataname + '_hr.csv',  dataset_config.dataname + '_rr.csv',
    #                       protocol=dataset.protocol, ks=config.evaluator.ks)

    evaluator = Evaluator(valid_input, valid_target, dataset_config['dataname'] + '_hr.csv',  dataset_config['dataname'] + '_rr.csv',
                          protocol=dataset.protocol, ks=config['evaluator']['ks'])
    model = model_base(dataset, hparams_cnfg, device)

    ret = model.fit(dataset, exp_config, evaluator=evaluator, early_stop=early_stop, loggers=[logger, csv_logger])
    print(ret['scores'])

    csv_logger.save()
    return {'HR@20': (ret['scores']['HR@20'], 0.0)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['LightGCN', 'BPRMF'], default='LightGCN',
                        help="model to choose")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.model == 'LightGCN':
        best_parameters, values, _experiment, _cur_model = optimize(
            parameters=[
                         {"name": "emb_dim", "type": "choice", "value_type": "int", "values": [20, 32, 64, 128]},
                         {"name": "num_layers", "type": "choice", "value_type": "int", "values": [2, 3, 4]},
                         {"name": "node_dropout", "type": "range", "value_type": "float", "bounds": [0.0, 0.1]},
                         {"name": "split", "type": "fixed", "value_type": "bool", "value": False},
                         {"name": "num_folds", "type": "fixed", "value_type": "int", "value": 100},
                         {"name": "graph_dir", "type": "fixed", "value_type": "str", "value": 'graph'},
                         {"name": "reg", "type": "fixed", "value_type": "float", "value": 0.0001},
                         {"name": "use_validation", "type": "fixed", "value_type": "bool", "value": True},
                     ],
            evaluation_function=train_with_conf,
            minimize=False,
            objective_name='HR@20',
            total_trials=5
        )

    else:
        best_parameters, values, _experiment, _cur_model = optimize(
            parameters=[
                {"name": "hidden_dim", "type": "choice", "value_type": "int", "values": [20, 30, 50, 70, 100, 120]},
                {"name": "pointwise", "type": "fixed", "value_type": "bool", "value": False},
                {"name": "loss_func", "type": "choice", "value_type": "str", "values": ['ce', 'mse']},
            ],
            evaluation_function=train_with_conf,
            minimize=False,
            objective_name='HR@20',
            total_trials=5
        )

    print('Final Train')
    best_parameters['use_validation'] = False
    train_with_conf(best_parameters)
