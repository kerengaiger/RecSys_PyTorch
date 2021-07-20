# Import packages
import os
import torch
import models

from utils.Params import Params
from utils.Dataset import Dataset
from utils.Logger import Logger
from utils.Evaluator import Evaluator
from utils.Trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='PureSVD')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./saves')
parser.add_argument('--conf_dir', type=str, default='./conf')
parser.add_argument('--seed', type=int, default=428)

conf = parser.parse_args()
model_conf = Params(os.path.join(conf.conf_dir, conf.model.lower() + '.json'))
model_conf.update_dict('exp_conf', conf.__dict__)

np.random.seed(conf.seed)
torch.random.manual_seed(conf.seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = Dataset(
    data_dir=conf.data_dir,
    data_name=model_conf.data_name,
    train_ratio=model_conf.train_ratio,
    device=device
)

log_dir = os.path.join('saves', conf.model)
logger = Logger(log_dir)
model_conf.save(os.path.join(logger.log_dir, 'config.json'))

eval_pos, eval_target = dataset.eval_data()
item_popularity = dataset.item_popularity
evaluator = Evaluator(eval_pos, eval_target, item_popularity, model_conf.top_k)

model_base = getattr(models, conf.model)
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

trainer.train()
=======
from data.dataset import UIRTDataset
from evaluation.evaluator import Evaluator
from experiment.early_stop import EarlyStop

from loggers import FileLogger, CSVLogger
from utils.general import make_log_dir, set_random_seed
from config import load_config
""" 
    Configurations
"""
config = load_config()

exp_config = config.experiment
gpu_id = exp_config.gpu
seed = exp_config.seed

dataset_config = config.dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    set_random_seed(seed)

    """ 
        Dataset
    """
    dataset = UIRTDataset(**dataset_config)

    # """ 
    #     Early stop
    # """
    # early_stop = EarlyStop(**config['EarlyStop'])


    """ 
        Model base class
    """
    model_name = config.experiment.model_name
    model_base = getattr(models, model_name)
    hparams = config.hparams

    """ 
        Logger
    """

    log_dir = make_log_dir(os.path.join(exp_config.save_dir, model_name))
    logger = FileLogger(log_dir)
    csv_logger = CSVLogger(log_dir)

    # Save log & dataset config.
    logger.info(config)
    logger.info(dataset)

    valid_input, valid_target = dataset.valid_input, dataset.valid_target
    evaluator = Evaluator(valid_input, valid_target, protocol=dataset.protocol, ks=config.evaluator.ks)

    model = model_base(dataset, hparams, device)

    ret = model.fit(dataset, exp_config, evaluator=evaluator, loggers=[logger, csv_logger])
    print(ret['scores'])
    
    csv_logger.save()
