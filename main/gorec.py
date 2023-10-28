import sys
sys.path.append('/home/cmm/Bai/test')
import argparse
import os
import torch
import time
import tool
from enviroment import Env
from loader.ae_loader import Loader4AE
from model.gorec_model import GoRec
from session.gorec_session import GoRec_session
def parse_args():   
    parser = argparse.ArgumentParser(description="GoRec")

    # ----------------------- File Identification
    parser.add_argument('--suffix', type=str, default='GoRec')

    # ----------------------- Device Setting
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='baby', help='baby, clothing, sports')
    parser.add_argument('--pre_model', type=str, default='vsgcl', help='mf, lgcn, vlgcn, lgcnf, vbpr, grcn, vbprm,simgcl,vsgcl')
    parser.add_argument('--uni_coeff', type=float, default= 5)
    # c 1    b 5    s 15
    parser.add_argument('--kl_coeff', type=float, default=10)
    # c 5000 b 10   s 5000

    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_start_epoch', type=int, default=0)

    # ------------------------ Training Setting
    parser.add_argument('--free_emb_dimension', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--encoder_layer', type=int, default=0)
    parser.add_argument('--decoder_layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--early_stop', type=int, default=15)
    parser.add_argument('--topk', type=str, default='[10, 20, 30, 40, 50]')
    parser.add_argument('--range4eval', type=str, default='cold')   
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--eva_interval', type=int, default=1)
    # parser.add_argument('--topk', type=str, default='[10]')

    # ----------------------- logger
    parser.add_argument('--log', type=int, default=0)
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)

    return parser.parse_args()


args = parse_args()
tool.cprint(f'---------- {args.suffix} ----------')
print(f'{args}')

# ----------------------------------- Env Init -----------------------------------------------------------

my_env = Env(args)
my_env.reset(args)
tool.cprint('Init Env')
# ----------------------------------- Dataset Init -----------------------------------------------------------

my_loader = Loader4AE(my_env)
tool.cprint('Init Dataset')

# ----------------------------------- Model Init -----------------------------------------------------------

my_model = GoRec(env=my_env, latent_dim=args.free_emb_dimension, 
                z_size=args.free_emb_dimension, si_dim=my_loader.feature.shape[1],
                training=True, encoder_layer=args.encoder_layer, decoder_layer=args.decoder_layer)
tool.cprint('Init Model')

# ----------------------------------- Session Init -----------------------------------------------------------

my_session = GoRec_session(my_env, my_model, my_loader)
tool.cprint('Init Session')

# ---------------------------------------- Main -----------------------------------------------------------

t = time.time()
my_session.train(args.epoch)
# my_session.save_memory()
my_env.close_env()
tool.cprint(f'training stage cost time: {time.time() - t}')

tool.cprint(f'--------- {args.suffix} best epoch {my_session.best_epoch}------------')
for top_k in eval(args.topk):
    tool.cprint(f'hr@{top_k} = {my_session.best_hr[top_k]:.5f}, recall@{top_k} = {my_session.best_recall[top_k]:.5f}, ndcg@{top_k} = {my_session.best_ndcg[top_k]:.5f}')
tool.cprint(f'--------- {args.suffix} test------------')
for top_k in eval(args.topk):
    tool.cprint(f'hr@{top_k} = {my_session.test_hr[top_k]:.5f}, recall@{top_k} = {my_session.test_recall[top_k]:.5f}, ndcg@{top_k} = {my_session.test_ndcg[top_k]:.5f}')
