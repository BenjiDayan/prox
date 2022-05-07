import argparse
import os
import pickle

# _____________ true directory to be filled _____________
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../GTA-1M/FPS-30')
parser.add_argument('--sequence_id', type=str, default='2020-05-21-14-06-15_resize')
parser.add_argument('--save_root', type=str, default='../proximity_map')

args = parser.parse_args()
# _____________ true directory to be filled _____________

