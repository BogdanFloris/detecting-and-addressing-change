"""
This class encloses different the constants contained in the constants library
"""
import os
import torch
from enum import Enum
from pathlib import Path
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer


class TransformerModel(Enum):
    BERT = (BertModel, BertTokenizer, "bert-base-uncased")
    SCIBERT = (
        BertModel,
        BertTokenizer,
        os.path.join(Path(__file__).parents[1], "assets/scibert_scivocab_uncased"),
    )
    DISTILBERT = (DistilBertModel, DistilBertTokenizer, "distilbert-base-uncased")


class Transformer:
    def __init__(self, model: TransformerModel = TransformerModel.BERT):
        # Unpack the tuple
        model_class, tokenizer_class, pretrained_weights = model.value

        # Initialize the model and the tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)

    def transform(self, text):
        """Transforms the given text using the initialized transformer

        Args:
            text (str): text to be transformed

        Returns:
            the transformed text
        """
        input_ids = torch.tensor(
            [self.tokenizer.encode(text, add_special_tokens=True, max_length=512)]
        )
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]

        return last_hidden_states
