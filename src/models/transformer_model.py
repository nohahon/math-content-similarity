from torch import nn
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig
from src.models.pooling import PoolingLayer
from typing import List, Dict, Optional
import pandas as pd
from src.data.dataset import TorchDataset


class Transformer(nn.Module):
    """Loads the embeddings model
    :param model_name_or_path
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_args,
    ):
        super(Transformer, self).__init__()

        self.model_args = model_args
        lm_config = AutoConfig.from_pretrained(model_name_or_path)
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.peft_config = LoraConfig(
            r=4,
            lora_alpha=32,
            lora_dropout=0.01,
        )
        self.language_model = self._load_model(
            model_name_or_path,
            lm_config,
            model_args=self.model_args,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
        )

        self.pooler_layer = PoolingLayer(
            pooling_mode="mean",
            word_embedding_dimension=lm_config.hidden_size,
        )
        self.config = self.language_model.config

    def _load_model(self, model_name_or_path, config, model_args):
        """Loads the transformer model"""
        if model_args['peft']:
            model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                quantization_config=self.quantization_config,
            )
            self.tokenizer.pad_token_id = 0
            return get_peft_model(model, self.peft_config)

        return AutoModel.from_pretrained(model_name_or_path, config=config)

    def forward(self, input_ids, attention_mask):
        """Returns token_embeddings, cls_token"""
        output = self.language_model(
            input_ids,
            attention_mask,
            return_dict=False,
        )
        features = {}
        features.update(
            {
                "token_embeddings": output[0],
                "attention_mask": attention_mask,
            },
        )
        features = self.pooler_layer(features)
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.language_model.config.hidden_size

    def tokenize(self, data: pd.DataFrame):
        tokenized = []

        def tokenize_single_(d):
            tokenized_anchor = self.tokenizer(
                d["anchor"],
                return_tensors="pt",
                padding="max_length",
                max_length=self.model_args['max_length'],
                truncation=True,
            )
            tokenized_rec = self.tokenizer(
                d["rec"],
                return_tensors="pt",
                padding="max_length",
                max_length=self.model_args['max_length'],
                truncation=True,
            )
            return {
                "input_ids": torch.cat(
                    [
                        tokenized_rec["input_ids"],
                        tokenized_anchor["input_ids"],
                    ],
                ),
                "attention_mask": torch.cat(
                    [
                        tokenized_rec["attention_mask"],
                        tokenized_anchor["attention_mask"],
                    ],
                ),
            }

        for idx, row in data.iterrows():
            tokenized.append(tokenize_single_(row))
        labels = data.label.tolist()
        return TorchDataset(tokenized, labels)
