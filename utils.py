# -*- coding:utf-8 -*-

import torch
import random
import numpy as np
from transformers import set_seed
import json
import os
from torch.utils.data import Dataset
import bitsandbytes as bnb
from tqdm import tqdm
import math

class QwenPromptDataSet(Dataset):
    """对话要素抽取所需的数据类"""

    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        """
        初始化函数
        Args:
            data_path: 数据路径
            tokenizer: 分词器
            max_len: 模型训练最大长度
            max_src_len: input的最大长度
            is_skip: 不符合长度标准的数据是否跳过
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_src_len = max_src_len
        self.is_skip = is_skip
        self.nl_tokens = self.tokenizer.encode("\n")
        self.all_data = self.load_data(data_path)

    def load_data(self, data_path):
        """
        加载原始数据，生成满足模型需求的数据，并剔除不达标的数据
        Args:
            data_path: 原始数据路径
        
        """
        self.all_data = []
        skip_data_number = 0
        
        # 遍历文件中的每一个样本
        with open(data_path, "r", encoding="utf-8")as f:
            for i, line in enumerate(f):
                sample = json.loads(line.strip())
                # 通过convert_feature函数将每一条数据进行索引化，生成模型所需要的input_ids和labels
                input_ids, labels, skip_flag = self.convert_feature(sample)
                # 跳过不符合标准的数据
                if self.is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids":input_ids, "labels":labels})
        print("the number of skipping data is {}, the proportion is {}".format(skip_data_number, skip_data_number / (
                len(self.all_data) + skip_data_number)))
        return self.all_data
    def _tokenize_str(self, role, content):
        """
        返回角色+内容文本和对应tokenizer的拼接
        Args:
            role: 角色
            content: 内容

        """
        role_content = f"{role}\n{content}"
        tokenizer_of_role_content = self.tokenizer.encode(role,allow_special=set()) + self.nl_tokens + self.tokenizer.encode(content, allow_special=set())
        return role_content, tokenizer_of_role_content
    
    def convert_feature(self, sample):
        """
        数据处理函数
        Args:
            sample: 包含提示词、输入内容、输出内容的字典， 格式为{"instruction":instruction, "input":input, "output":output}

        """
        skip_flag = False
        im_strat_tokens = [self.tokenizer.im_start_id]
        im_end_tokens = [self.tokenizer.im_end_id]
        # 构造qwen模型所需要的系统指令内容
        sys_prompt = "You are a helpful assistant."
        system_text, system_tokens_part = self._tokenize_str("system", sys_prompt)
        system_tokens = im_strat_tokens + system_tokens_part + im_end_tokens

        input_ids = []
        labels = []
        # 构建用户输入内容
        prompt_ids = im_strat_tokens + self._tokenize_str("user", sample["instruction"]+ sample["input"])[1] + im_end_tokens
        # 当用户输入内容长度超过最大长度时，进行向前截断，并生成对应的label
        if len(prompt_ids) > self.max_src_len:
            input_ids = self.nl_tokens + prompt_ids[:self.max_src_len - 1] + [prompt_ids[-1]]
            labels = [-100] * (len(input_ids))
            skip_flag = True
        else:
            input_ids.extend(self.nl_tokens + prompt_ids)
            labels.extend([-100] * (len(prompt_ids) + len(self.nl_tokens)))
        assert len(input_ids) == len(labels)
        # 构建模型输出内容
        output_id = im_strat_tokens + self._tokenize_str("assistant", sample["output"])[1] + im_end_tokens
        # 当模型输出内容长度超过最大长度时， 向前截断
        max_tgt_len = self.max_len - len(input_ids) - len(sys_prompt)
        if len(output_id) > max_tgt_len:
            output_id = output_id[:max_tgt_len - 1] + [output_id[-1]]
            skip_flag = True
        # 将系统指令、 用户输入、 模型输出进行拼接， 构建完整的模型训练所需数据
        input_ids = system_tokens + input_ids + self.nl_tokens + output_id
        labels = [-100] * len(system_tokens) + labels + [-100] * (1 + len(self.nl_tokens)) + output_id[1:]
        
        assert len(input_ids) == len(labels)
        assert len(input_ids) <= self.max_len

        return input_ids, labels, skip_flag
