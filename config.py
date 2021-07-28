from typing import List, Tuple
from dataclasses import dataclass, field

from omegaconf import OmegaConf

@dataclass
class DatasetConfig:
    data_path:str='datasets/ml-100k/u.data'
    dataname:str='ml-1m'
    separator:str='\t'
    binarize_threshold:float=4.0
    implicit:bool=True
    min_usr_len:int=1
    max_usr_len:int=60
    min_items_cnt:int=10
    max_items_cnt:int=10000
    final_usr_len:int=4
    min_item_per_user:int=10
    min_user_per_item:int=1
    max_item_per_user:int=60
    max_user_per_item:int=100000

    protocol:str='leave_one_out' # holdout, leave_one_out
    generalization:str='weak' # weak/strong
    holdout_users:int=1

    valid_ratio:float=0.0
    test_ratio:float=0.0
    leave_k:int=1
    split_random:bool=False

@dataclass
class EvaluatorConfig:
    ks:List[int] = field(default_factory=lambda: [20, 10, 5])

@dataclass
class EarlyStopConfig:
    early_stop:int=25
    early_stop_measure:str='loss'

@dataclass
class ExperimentConfig:
    debug:bool=False
    save_dir:str='saves'
    num_epochs:int=300
    batch_size:int=256
    verbose:int=0
    print_step:int=1
    test_step:int=1
    test_from:int=1
    model_name:str='LightGCN'
    num_exp:int=5
    seed:int=2020
    gpu:int=0

def load_config():
    dataset_config = OmegaConf.structured({'dataset' :DatasetConfig})
    evaluator_config = OmegaConf.structured({'evaluator': EvaluatorConfig})
    early_stop_config = OmegaConf.structured({'early_stop': EarlyStopConfig})
    experiment_config = OmegaConf.structured({'experiment': ExperimentConfig})
    
    model_name = experiment_config.experiment.model_name
    # model_config = OmegaConf.structured({'hparams': OmegaConf.load(f"conf/{model_name}.yaml")})
    model_config = OmegaConf.structured(OmegaConf.load(f"conf/{model_name}.yaml"))
    
    config = OmegaConf.merge(dataset_config, evaluator_config, early_stop_config, experiment_config, model_config)
    return config

if __name__ == '__main__':
    config = load_config()
    print(config)