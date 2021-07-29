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


def train_with_conf(hparams_cnfg):
    config = load_config()

    exp_config = config.experiment
    gpu_id = exp_config.gpu
    seed = exp_config.seed

    dataset_config = config.dataset

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
    early_stop = EarlyStop(**config['EarlyStop'])

    """ 
        Model base class
    """
    model_name = config.experiment.model_name
    model_base = getattr(models, model_name)
    log_dir = make_log_dir(os.path.join(exp_config.save_dir, model_name))
    logger = FileLogger(log_dir)
    csv_logger = CSVLogger(log_dir)

    # Save log & dataset config.
    logger.info(config)
    logger.info(dataset)

    valid_input, valid_target = dataset.valid_input, dataset.valid_target
    evaluator = Evaluator(valid_input, valid_target, protocol=dataset.protocol, ks=config.evaluator.ks)

    model = model_base(dataset, hparams_cnfg, device)

    ret = model.fit(dataset, exp_config, evaluator=evaluator,early_stop=early_stop, loggers=[logger, csv_logger])
    # print(ret['scores'])

    csv_logger.save()
    return {'validation loss': ret['scores']}


if __name__ == '__main__':
    best_parameters, values, _experiment, _cur_model = optimize(
                parameters=[
                    {"name": "emb_dim", "type": "choice", "value_type": "int", "values": [20, 32, 64, 128]},
                    {"name": "num_layers", "type": "choice", "value_type": "int", "values": [2, 4, 6]},
                    {"name": "node_dropout", "type": "range", "value_type": "float", "bounds": [0.2, 0.5]},
                    {"name": "split", "type": "fixed", "value_type": "bool", "value": False},
                    {"name": "num_folds", "type": "fixed", "value_type": "int", "value": 100},
                    {"name": "graph_dir", "type": "fixed", "value_type": "str", "value": 'graph'},
                    {"name": "reg", "type": "fixed", "value_type": "float", "value": 0.0001},
                    {"name": "use_validation", "type": "fixed", "value_type": "bool", "value": True},


                ],
                evaluation_function=train_with_conf,
                minimize=True,
                objective_name='validation loss',
                total_trials=5
            )
    print('Final Train')
    best_parameters['use_validation'] = False
    train_with_conf(best_parameters)