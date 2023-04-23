#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2023-04-24 00:24:38
LastEditTime: 2023-04-24 00:24:41
LastEditors: Kun
Description: 
FilePath: /Alpaca-LoRA-RLHF-PyTorch/data_loader/sft_dataloader.py
'''

from datasets import load_dataset
from utils.prompter import Prompter


class SFTDataLoader(object):
    def __init__(self, data_path, cutoff_len, val_set_size, train_on_inputs, add_eos_token, prompt_template_name, tokenizer) -> None:
        super(SFTDataLoader, self).__init__()

        self.data_path = data_path
        self.cutoff_len = cutoff_len
        self.val_set_size = val_set_size
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token

        self.prompter = Prompter(prompt_template_name)

        self.tokenizer = tokenizer

    def tokenize(self, prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(
                user_prompt, add_eos_token=self.add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def load_data(self):
        if self.data_path.endswith(".json") or self.data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=self.data_path)
        else:
            data = load_dataset(self.data_path)

        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(
                    self.generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(
                    self.generate_and_tokenize_prompt)
            )
            train_data = train_data.select(range(1000))
            val_data = val_data.select(range(100))
        else:
            train_data = data["train"].shuffle().map(
                self.generate_and_tokenize_prompt)
            val_data = None
            train_data = train_data.select(range(1000))

        return train_data, val_data
