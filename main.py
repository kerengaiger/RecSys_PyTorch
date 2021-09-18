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
import optuna


def train_with_conf(conf):
    model_conf = Params(os.path.join(conf['conf_dir'], f'{conf["model"].lower()}_{conf["data_name"]}.json'))
    for k in conf.keys():
        model_conf.update_dict(k, conf[k])

    np.random.seed(conf['seed'])
    torch.random.manual_seed(conf['seed'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

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
    print(eval_target)
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
    hr_k = evaluator.hit_ratio_k(model, model_conf.test_batch_size, model_conf.data_name, conf['data_dir'],
                                 conf['save_dir'])
    mrr_k = evaluator.mrr_k(model, model_conf.test_batch_size, model_conf.data_name, conf['data_dir'],
                            conf['save_dir'])
    print(f'hr_{evaluator.max_k}:{hr_k}')
    print(f'mrr_{evaluator.max_k}:{mrr_k}')
    if not conf['early_stop']:
        torch.save(model, pathlib.Path(conf['save_dir'], model_conf.data_name + '_bpr.pt'))
        pickle.dump(model_conf, open(pathlib.Path(conf['save_dir'], model_conf.data_name + '_cnfg.pkl'), 'wb'))
    return best_score, best_epoch


class Objective:
    def __init__(self):
        self.best_epoch = None

    def __call__(self, trial):
        cnfg = {}
        args = parse_args()
        cnfg['data_name'] = args.data_name
        cnfg['conf_dir'] = args.conf_dir
        cnfg['model'] = args.model
        cnfg['seed'] = args.seed
        cnfg['data_dir'] = args.data_dir
        cnfg['save_dir'] = args.save_dir
        cnfg['use_validation'] = True
        cnfg['early_stop'] = True
        cnfg['hidden_dim'] = trial.suggest_int("hidden_dim", 10, 100, step=4)
        cnfg['learning_rate'] = trial.suggest_float("learning_rate", 5e-3, 1e-1)
        cnfg['batch_size'] = trial.suggest_categorical("batch_size", [128, 256, 500, 1024])
        cnfg['loss_func'] = trial.suggest_categorical("loss_func", ['ce', 'mse'])
        validation_loss, best_epoch = train_with_conf(cnfg)
        self.best_epoch = best_epoch
        return validation_loss

    def callback(self, study, trial):
        args = parse_args()
        if study.best_trial == trial:
            best_cnfg = trial.params
            best_cnfg['best_epoch'] = self.best_epoch
            pickle.dump(best_cnfg, open(pathlib.Path(args.save_dir, args.data_name + '_cnfg.pkl'), 'wb'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='EASE')
    parser.add_argument('--data_name', type=str, default='ml-1m')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--save_dir', type=str, default='./saves')
    parser.add_argument('--conf_dir', type=str, default='./conf')
    parser.add_argument('--seed', type=int, default=428)

    return parser.parse_args()


def main():
    args = parse_args()
    objective = Objective()
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize"
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
    best_parameters = pickle.load(open(pathlib.Path(args.save_dir, args.data_name + '_cnfg.pkl'), 'rb'))
    best_parameters['use_validation'] = False
    best_parameters['early_stop'] = False
    best_parameters['num_epochs'] = int(best_parameters['best_epoch'])
    train_with_conf(best_parameters)


if __name__ == '__main__':
    main()
