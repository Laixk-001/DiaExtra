# -*- coding:utf-8 -*-

import torch
from qwen1_8.modeling_qwen import QWenLMHeadModel
from qwen1_8.tokenization_qwen import QWenTokenizer
import argparse
from peft import PeftModel

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device')
    parser.add_argument('--ori_model_dir', default="Qwen-1_8-chat/",type=str, help='model path')
    parser.add_argument('--model_dir', default="/root/auto-fs/DiaExtra/output_dir_qlora/",type=str,help='qlora path')