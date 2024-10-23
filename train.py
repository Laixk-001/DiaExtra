# -*- coding:utf-8 -*-

import argparse
import json
import math
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from transformers import BitsAndBytesConfig
from utils import print_trainable_parameters, print_rank_0, to_device, set_random_seed, save_model, DataCollator, \
    find_all_linear_names, evaluation
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen1_8.modeling_qwen import QWenLMHeadModel
from qwen1_8.tokenization_qwen import QWenTokenizer
from qwen1_8.configuration_qwen import QWenConfig
from utils import QwenPromptDataSet
import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter
