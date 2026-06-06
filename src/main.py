"""
main.py — DiffuRec + Category Embedding + Taxonomy Loss (Combined)
==================================================================
Thêm các argument mới:
  --cat_alpha        : trọng số category embedding (default 0.3)
  --alpha_taxonomy   : trọng số taxonomy loss (default 0.1)
  --beta_taxonomy    : tỉ lệ Focal vs Triplet (default 0.6)
  --margin_taxonomy  : margin Triplet Loss (default 0.8)
  --loss_scale       : scale taxonomy loss (default 10.0)
  --warmup_epochs    : số epoch warm-up (default 10)
  --rampup_epochs    : số epoch ramp-up (default 10)
"""

import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import pickle
from utils import Data_Train, Data_Val, Data_Test, Data_CHLS
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train, LSHT_inference
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',    default='amazon_beauty')
parser.add_argument('--log_file',   default='log/')
parser.add_argument('--random_seed',type=int, default=1997)
parser.add_argument('--max_len',    type=int, default=50)
parser.add_argument('--device',     type=str, default='cuda', choices=['cpu','cuda'])
parser.add_argument('--num_gpu',    type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--hidden_size',type=int, default=128)
parser.add_argument('--dropout',    type=float, default=0.1)
parser.add_argument('--emb_dropout',type=float, default=0.3)
parser.add_argument('--hidden_act', default='gelu')
parser.add_argument('--num_blocks', type=int, default=4)
parser.add_argument('--epochs',     type=int, default=500)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma',      type=float, default=0.1)
parser.add_argument('--metric_ks',  nargs='+', type=int, default=[5,10,20])
parser.add_argument('--optimizer',  type=str, default='Adam')
parser.add_argument('--lr',         type=float, default=0.001)
parser.add_argument('--loss_lambda',type=float, default=0.001)
parser.add_argument('--weight_decay',type=float, default=0)
parser.add_argument('--momentum',   type=float, default=None)
parser.add_argument('--schedule_sampler_name', default='lossaware')
parser.add_argument('--diffusion_steps', type=int, default=32)
parser.add_argument('--lambda_uncertainty', type=float, default=0.001)
parser.add_argument('--noise_schedule', default='trunc_lin')
parser.add_argument('--rescale_timesteps', default=True)
parser.add_argument('--eval_interval', type=int, default=20)
parser.add_argument('--patience',   type=int, default=5)
parser.add_argument('--description',default='Combined_CatEmb_TaxLoss')
parser.add_argument('--long_head',  default=False)
parser.add_argument('--diversity_measure', default=False)
parser.add_argument('--epoch_time_avg', default=False)

# ── Taxonomy arguments ────────────────────────────────────────────────────
parser.add_argument('--cat_alpha',       type=float, default=0.05,
                    help='Trọng số category embedding trong input')
parser.add_argument('--alpha_taxonomy',  type=float, default=0.3,
                    help='Trọng số tổng thể taxonomy loss')
parser.add_argument('--beta_taxonomy',   type=float, default=0.6,
                    help='Tỉ lệ Focal vs Triplet (0=full Triplet, 1=full Focal)')
parser.add_argument('--margin_taxonomy', type=float, default=0.8,
                    help='Margin cho Triplet Loss')
parser.add_argument('--loss_scale',      type=float, default=10.0,
                    help='Scale taxonomy loss về cùng magnitude CE')
parser.add_argument('--warmup_epochs',   type=int,   default=10,
                    help='Số epoch chỉ train CE, chưa bật taxonomy')
parser.add_argument('--rampup_epochs',   type=int,   default=10,
                    help='Số epoch tăng dần taxonomy weight 0 → alpha')

args = parser.parse_args()
print(args)

os.makedirs(args.log_file, exist_ok=True)
os.makedirs(args.log_file + args.dataset, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=args.log_file + args.dataset + '/'
             + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)
logger.info(args)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, item_num):
    args.item_num = item_num
    return args


def cold_hot_long_short(data_raw, dataset_name):
    item_list, len_list, target_item = [], [], []
    for id_temp in data_raw['train']:
        temp_list = (data_raw['train'][id_temp]
                     + data_raw['val'][id_temp]
                     + data_raw['test'][id_temp])
        len_list.append(len(temp_list))
        target_item.append(data_raw['test'][id_temp][0])
        item_list += temp_list
    item_num_count = Counter(item_list)
    split_num = np.percentile(list(item_num_count.values()), 80)
    cold_item, hot_item = [], []
    for k, v in item_num_count.items():
        (hot_item if v >= split_num else cold_item).append(k)
    cold_list, hot_list = [], []
    for id_temp, item_temp in enumerate(data_raw['test'].values()):
        offset = id_temp + 1 if dataset_name == 'ml-1m' else id_temp
        seq = (data_raw['train'][offset]
               + data_raw['val'][offset]
               + data_raw['test'][offset])
        if item_temp[0] in hot_item:
            hot_list.append(seq)
        else:
            cold_list.append(seq)
    cold_hot_dict = {'hot': hot_list, 'cold': cold_list}
    p20, p40, p60, p80 = [np.percentile(len_list, p) for p in [20,40,60,80]]
    len_seq_dict = {'short':[], 'mid_short':[], 'mid':[], 'mid_long':[], 'long':[]}
    for id_temp, lt in enumerate(len_list):
        offset = id_temp + 1 if dataset_name == 'ml-1m' else id_temp
        seq = (data_raw['train'][offset]
               + data_raw['val'][offset]
               + data_raw['test'][offset])
        if lt <= p20:                 len_seq_dict['short'].append(seq)
        elif lt <= p40:               len_seq_dict['mid_short'].append(seq)
        elif lt <= p60:               len_seq_dict['mid'].append(seq)
        elif lt <= p80:               len_seq_dict['mid_long'].append(seq)
        else:                         len_seq_dict['long'].append(seq)
    return (cold_hot_dict, len_seq_dict, split_num,
            [p20, p40, p60, p80], len_list, list(item_num_count.values()))


def main(args):
    fix_random_seed_as(args.random_seed)

    path_data = '../datasets/data/' + args.dataset + '/dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)

    # ── Load category map ─────────────────────────────────────────────────
    category_map = data_raw.get('category_map', None)
    category_num = 0
    if category_map is not None:
        all_cats = set()
        for v in category_map.values():
            if isinstance(v, list):
                all_cats.update(v)
            else:
                all_cats.add(v)
        category_num = max(all_cats) if all_cats else 0
        print(f">>> Category map loaded | num_categories={category_num}")
        print(f">>> CatEmb  : cat_alpha={args.cat_alpha}")
        print(f">>> TaxLoss : alpha={args.alpha_taxonomy} | "
              f"beta={args.beta_taxonomy} | margin={args.margin_taxonomy} | "
              f"scale={args.loss_scale}")
        print(f">>> Warmup  : warmup={args.warmup_epochs} | "
              f"rampup={args.rampup_epochs}")
    else:
        print(">>> No category_map → Baseline DiffuRec")

    args = item_num_create(args, len(data_raw['smap']))

    tra_data  = Data_Train(data_raw['train'], args)
    val_data  = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'],
                          data_raw['test'], args)

    tra_data_loader  = tra_data.get_pytorch_dataloaders()
    val_data_loader  = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()

    diffu_rec = create_model_diffu(args)

    # Tạo Combined model
    model = Att_Diffuse_model(
        diffu_rec, args,
        category_map = category_map,
        category_num = category_num,
    )

    best_model, test_results = model_train(
        tra_data_loader, val_data_loader, test_data_loader,
        model, args, logger
    )

    if args.long_head:
        (cold_hot_dict, len_seq_dict, split_hotcold,
         split_length, list_len, list_num) = cold_hot_long_short(
            data_raw, args.dataset
        )
        for label, data_list in [
            ('Cold item',  cold_hot_dict['cold']),
            ('Hot item',   cold_hot_dict['hot']),
            ('Short',      len_seq_dict['short']),
            ('Mid_short',  len_seq_dict['mid_short']),
            ('Mid',        len_seq_dict['mid']),
            ('Mid_long',   len_seq_dict['mid_long']),
            ('Long',       len_seq_dict['long']),
        ]:
            loader = Data_CHLS(data_list, args).get_pytorch_dataloaders()
            print(f'--------------{label}-----------------------')
            LSHT_inference(best_model, args, loader)


if __name__ == '__main__':
    main(args)
