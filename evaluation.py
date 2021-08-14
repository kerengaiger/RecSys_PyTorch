import pathlib
import torch
from utils.Params import Params
from utils.Dataset import Dataset
from utils.Evaluator import Evaluator
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='EASE')
parser.add_argument('--data_name', type=str, default='ml-1m')
parser.add_argument('--tune', action='store_true')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./saves')
parser.add_argument('--conf_dir', type=str, default='./conf')
parser.add_argument('--seed', type=int, default=428)
conf = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
model_conf = Params(os.path.join(conf.conf_dir, f'{conf.model.lower()}_{conf.data_name}.json'))

dataset = Dataset(
    data_dir=conf.data_dir,
    data_name=model_conf.data_name,
    train_ratio=0.8,
    device=device,
    min_usr_len=model_conf.min_usr_len,
    max_usr_len=model_conf.max_usr_len,
    min_item_cnt=model_conf.min_item_cnt,
    max_item_cnt=model_conf.max_item_cnt,
    fin_min_usr_len=model_conf.fin_min_usr_len,
    pos_thresh=model_conf.pos_thresh,
    use_validation=False
)
eval_pos, eval_target = dataset.eval_data()
item_popularity = dataset.item_popularity
evaluator = Evaluator(eval_pos, eval_target, item_popularity, model_conf.top_k)
model = torch.load(pathlib.Path(conf.save_dir, model_conf.data_name + '_bpr.pt'))
mrr_k = evaluator.mrr_k(model, model_conf.test_batch_size, model_conf.data_name, conf.data_dir,
                            conf.save_dir)

