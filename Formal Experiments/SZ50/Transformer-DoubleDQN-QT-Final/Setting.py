import argparse
import torch
import numpy as np
import random
parser = argparse.ArgumentParser()
parser.add_argument("--history_data_len",default=20,type=int,help="Historical days as seen by the model")
parser.add_argument('--window',default=16,type=int,help='The size of the sliding window when looking for an inflection point')
parser.add_argument('--ADayTime',default=16,type=int,help='The number of bars contained in the time of day')

parser.add_argument("--start_timesteps", default=500, type=int, help='how many step random policy run')
parser.add_argument("--max_timesteps", default=2000, type=float, help='max_timesteps')
parser.add_argument("--expl_noise", default=0.3, type=float, help='Gaussian exploration')
parser.add_argument("--batch_size", default=128, type=int, help='Batch size')
parser.add_argument("--GAMMA", default=0.99, type=float, help='Discount')
parser.add_argument("--tau", default=0.05, type=float, help='DDPG update rate')
parser.add_argument("--policy_noise", default=0.2, type=float, help='Noise to target policy during critic update')
parser.add_argument("--noise_clip", default=0.5, type=float, help='Range to clip target policy noise')
parser.add_argument("--policy_freq", default=2, type=int, help=' Frequency of delayed policy updates')

arg = parser.parse_args()

arg.history_data_len

SEED = 1

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)