import os
import torch

import models
from data.dataset import UIRTDataset
from evaluation.evaluator import Evaluator
from experiment.early_stop import EarlyStop

from loggers import FileLogger, CSVLogger
from utils.general import make_log_dir, set_random_seed
from config import load_config

from
if __name__ == '__main__':
    best_parameters, values, _experiment, _cur_model = optimize(
                parameters=[
                    {"name": "learning_rate", "type": "range", "value_type": "float", "bounds": [5e-3, 1e-1]},
                    {"name": "hidden_dim", "type": "choice", "value_type": "int", "values": [12, 17, 20, 25, 30, 50, 100]},
                    {"name": "batch_size", "type": "choice", "value_type": "int", "values": [100, 200, 1024, 4096, 3200, 6000]},
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