from evaluation.evaluator import Evaluator
import os
import torch

from data.dataset import UIRTDataset
from config import load_config

if __name__ == '__main__':
    config = load_config()
    model_name = config.experiment.model_name
    dataset_config = config.dataset
    dataset = UIRTDataset(**dataset_config)
    exp_config = config.experiment
    valid_input, valid_target = dataset.valid_input, dataset.valid_target
    evaluator = Evaluator(valid_input, valid_target, f'{dataset_config.dataname}_{model_name}_preds.csv',
                          protocol=dataset.protocol, ks=config.evaluator.ks, usermap_file=dataset._user2id_file,
                          is_final_train=True, itemmap_file=dataset._item2id_file)
    model = torch.load(os.path.join(exp_config.save_dir, f'{dataset_config.dataname}_{model_name}.pt'))
    evaluator.evaluate(model)