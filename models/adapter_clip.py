from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .clip import clip_loader
from .clip.tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class AdapterCLIP(nn.Module):

    def __init__(self,
                 model_name,
                 peft_method='adapter',
                 peft_encoder='both',
                 device='cpu'):
        super(AdapterCLIP, self).__init__()
        self.device = device

        design_details = {
            'method': peft_method,
            'peft_encoder': peft_encoder,
            'ffn_num': 64,
            'lora_alpha': 1,
            'lora_r': 4
        }

        self.model = clip_loader.load(model_name,
                                      device=self.device,
                                      jit=False,
                                      design_details=design_details)

        self.text_tokens = None
        self.current_class_names = []

        self.prompt_template = "a bad photo of a {}."

    def labels_tokenize(self,
                        labels: Union[str, List[str]],
                        context_length: int = 77) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        labels : Union[str, List[str]]
            An input string or a list of labels to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(labels, str):
            labels = [labels]

        texts = [self.prompt_template.format(c) for c in labels]

        sot_token = _tokenizer.encoder["<start_of_text>"]
        eot_token = _tokenizer.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                      for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:  # Truncate
                tokens = tokens[:context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result.to(self.device)

    def update_class_names(self, new_class_names):
        _num = 0
        for c in new_class_names:
            if c not in self.current_class_names:
                self.current_class_names.append(c)
                _num += 1
        if _num > 0:
            self.text_tokens = self.labels_tokenize(self.current_class_names)
        return self.text_tokens

    def forward(self, image, text_tokens=None):
        if text_tokens is None:
            text_tokens = self.text_tokens
        logits_per_image, _, image_features, text_features = self.model(
            image, text_tokens)
        probs = logits_per_image.softmax(dim=-1)
        return probs, image_features, text_features
