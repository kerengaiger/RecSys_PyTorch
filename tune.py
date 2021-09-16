import os
import torch

import models
from data.dataset import UIRTDataset
from evaluation.evaluator import Evaluator
from experiment.early_stop import EarlyStop

from loggers import FileLogger, CSVLogger
from utils.general import make_log_dir, set_random_seed
from config import load_config

import optuna
import pickle
import argparse


def train_with_conf(hparams_cnfg, trial=None):
    is_final_train = not hparams_cnfg['use_validation']
    config = load_config()
    exp_config = config.experiment
    gpu_id = exp_config.gpu
    seed = exp_config.seed

    dataset_config = config.dataset
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
    early_stop_config = config.early_stop
    early_stop_config['is_final_train'] = is_final_train
    early_stop = EarlyStop(**early_stop_config)


    """ 
        Model base class
    """
    model_name = config.experiment.model_name
    model_base = getattr(models, model_name)
    log_dir = make_log_dir(os.path.join(exp_config.save_dir, model_name))
    logger = FileLogger(log_dir)
    csv_logger = CSVLogger(log_dir)
    config.hparams = hparams_cnfg
  
    # Save log & dataset config.
    logger.info(config)
    logger.info(dataset)

    valid_input, valid_target = dataset.valid_input, dataset.valid_target
    evaluator = Evaluator(valid_input, valid_target, f'{dataset_config.dataname}_{model_name}_preds.csv',
                          protocol=dataset.protocol, ks=config.evaluator.ks, usermap_file=dataset._user2id_file,
                          is_final_train=is_final_train, itemmap_file=dataset._item2id_file)

    model = model_base(dataset, hparams_cnfg, device)

    ret = model.fit(dataset, exp_config, evaluator=evaluator, early_stop=early_stop, loggers=[logger, csv_logger])
    print(ret['scores'])

    csv_logger.save()
    if is_final_train:
        torch.save(model, os.path.join(exp_config.save_dir, f'{dataset_config.dataname}_{model_name}.pt'))
    return ret['scores']['HR@20']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['LightGCN', 'BPRMF'], default='LightGCN',
                        help="model to choose")
    parser.add_argument('--trials', type=int, default=50, help="trials")
    return parser.parse_args()


class Objective:
    def __init__(self, model):
        self.model = model
        config = load_config()
        self.exp_config = config.experiment
        self.dataset_config = config.dataset

    def __call__(self, trial):
        cnfg = {}
        if self.model == 'LightGCN':
            cnfg['split'] = trial.suggest_categorical("split", [False])
            cnfg['num_folds'] = trial.suggest_int("num_folds", 100, 100)
            cnfg['graph_dir'] = trial.suggest_categorical("graph_dir", ['graph'])
            cnfg['reg'] = trial.suggest_float("reg", 0.0001, 0.0001)
            cnfg['node_dropout'] = trial.suggest_float("node_dropout", 0.1, 0.6)
            cnfg['emb_dim'] = trial.suggest_int("emb_dim", 10, 100, step=4)
            cnfg['num_layers'] = trial.suggest_int("num_layers", 2, 5, step=1)
            cnfg['use_validation'] = True
        else:
            cnfg['pointwise'] = trial.suggest_categorical("split", [False])
            cnfg['hidden_dim'] = trial.suggest_int("hidden_dim", 10, 100, step=4)
            cnfg['lr'] = trial.suggest_float("lr", 5e-3, 1e-1)
            cnfg['loss_func'] = trial.suggest_categorical("loss_func", ['ce', 'mse'])
            cnfg['use_validation'] = True
        hr_20 = train_with_conf(cnfg, trial)
        return hr_20

    def callback(self, study, trial):
        if study.best_trial == trial:
            best_cnfg = trial.params
            pickle.dump(best_cnfg, open(os.path.join(self.exp_config.save_dir,
                                                     f'{self.dataset_config.dataname}_{self.model}_cnfg_pkl', 'wb')))


def main():
    args = parse_args()
    objective = Objective(model=args.model)

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize"
    )
    study.optimize(objective, n_trials=args.trials, callbacks=[objective.callback])

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)

    print('Final Train')
    best_parameters = best_trial.params
    best_parameters['use_validation'] = False
    train_with_conf(best_parameters)


if __name__ == '__main__':
    main()
