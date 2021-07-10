import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
args.device = torch.device('cuda', args.local_rank)
# 初始化分布式环境，主要用来帮助进程间通信
torch.distributed.init_process_group(backend='nccl')