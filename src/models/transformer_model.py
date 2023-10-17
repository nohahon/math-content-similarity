from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from src.models.pooling import PoolingLayer
import json
from typing import List, Dict, Optional, Union, Tuple
import os


class Transformer(nn.Module):
    """Loads the embeddings model
    :param model_name_or_path
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: Optional[int] = None,
        model_args: Dict = {},
    ):
        super(Transformer, self).__init__()

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args)
        self.language_model = self._load_model(
            model_name_or_path,
            config,
            **model_args,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.pooler_layer = PoolingLayer(
            pooling_mode="mean",
            word_embedding_dimension=config.hidden_size,
        )

        self.max_seq_length = max_seq_length

    def _load_model(self, model_name_or_path, config, **model_args):
        """Loads the transformer model"""
        # if load_llama -> self.load_llama else:
        return AutoModel.from_pretrained(
            model_name_or_path,
            config=config,
            **model_args,
        )

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = self.language_model(
            **trans_features,
            return_dict=False,
        )
        output_tokens = output_states[0]

        features.update(
            {
                "token_embeddings": output_tokens,
                "attention_mask": features["attention_mask"],
            },
        )

        if self.language_model.config.output_hidden_states:
            all_layer_idx = 2
            if (
                len(output_states) < 3
            ):  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        features = self.pooler_layer(features)

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.language_model.config.hidden_size

    def tokenize(
        self,
        texts: Union[List[str], List[Dict], List[Tuple[str, str]]],
    ):
        return None

    def save(self, output_path: str):
        self.language_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(
            os.path.join(output_path, "sentence_bert_config.json"),
            "w",
        ) as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)
