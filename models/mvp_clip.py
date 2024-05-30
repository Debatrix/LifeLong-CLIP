import logging
from typing import List, TypeVar, Iterable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip import clip_loader
from .clip.tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

logger = logging.getLogger()

T = TypeVar('T', bound='nn.Module')


class CLIP_MVP(nn.Module):

    def __init__(self,
                 pos_g_prompt: Iterable[int] = (0, 1),
                 len_g_prompt: int = 5,
                 pos_e_prompt: Iterable[int] = (2, 3, 4),
                 len_e_prompt: int = 20,
                 selection_size: int = 1,
                 prompt_func: str = 'prompt_tuning',
                 task_num: int = 10,
                 num_classes: int = 100,
                 lambd: float = 1.0,
                 use_mask: bool = True,
                 use_contrastiv: bool = False,
                 use_last_layer: bool = True,
                 model_name='ViT-B/16',
                 device='cpu',
                 **kwargs):

        super().__init__()

        self.features = torch.empty(0)
        self.keys = torch.empty(0)

        self.lambd = lambd
        self.class_num = num_classes
        self.task_num = task_num
        self.use_mask = use_mask
        self.use_contrastiv = use_contrastiv
        self.use_last_layer = use_last_layer
        self.selection_size = selection_size
        self.device = device

        # CLIP setting
        model = clip_loader.load(model_name,
                                 device=self.device,
                                 jit=False,
                                 design_details={
                                     'method': 'mvp',
                                     'peft_encoder': 'none',
                                 })

        self.add_module('backbone', model)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.prompt_template = "a bad photo of a {}."
        self.text_tokens = None
        self.current_class_names = []

        embed_dim = self.backbone.visual.conv1.weight.shape[0]

        # prompt
        self.register_buffer('pos_g_prompt',
                             torch.tensor(pos_g_prompt, dtype=torch.int64))
        self.register_buffer('pos_e_prompt',
                             torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.zeros(1))

        self.len_g_prompt = len_g_prompt
        self.len_e_prompt = len_e_prompt
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0
        g_pool = 1
        e_pool = task_num

        self.register_buffer('count', torch.zeros(e_pool))
        self.key = nn.Parameter(torch.randn(e_pool, embed_dim))
        self.mask = nn.Parameter(torch.zeros(e_pool, self.class_num) - 1)

        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_size = 1 * self.g_length * self.len_g_prompt
            self.e_size = 1 * self.e_length * self.len_e_prompt
            self.g_prompts = nn.Parameter(
                torch.randn(g_pool, self.g_size, embed_dim))
            self.e_prompts = nn.Parameter(
                torch.randn(e_pool, self.e_size, embed_dim))

        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_size = 2 * self.g_length * self.len_g_prompt
            self.e_size = 2 * self.e_length * self.len_e_prompt
            self.g_prompts = nn.Parameter(
                torch.randn(g_pool, self.g_size, embed_dim))
            self.e_prompts = nn.Parameter(
                torch.randn(e_pool, self.e_size, embed_dim))

        self.exposed_classes = 0

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

    @torch.no_grad()
    def set_exposed_classes(self, classes_names):
        self.exposed_classes = len(classes_names)

        # update class names
        _new_class_flag = False
        for c in classes_names:
            if c not in self.current_class_names:
                self.current_class_names.append(c)
                _new_class_flag = True

        # update text tokens
        if _new_class_flag:
            self.text_tokens = self.labels_tokenize(self.current_class_names)
        self.text_tokens = self.text_tokens.to(self.device)

        return self.text_tokens

    def prompt_tuning(self, x: torch.Tensor, g_prompt: torch.Tensor,
                      e_prompt: torch.Tensor, **kwargs):

        N, B, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, -1, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, -1, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.visual.transformer.resblocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                x = torch.cat((x, g_prompt[:, pos_g].permute(1, 0, 2)), dim=0)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e].permute(1, 0, 2)), dim=0)
            x = block(x)
            x = x[:N, :, :]
        return x

    def prefix_tuning(self, x: torch.Tensor, g_prompt: torch.Tensor,
                      e_prompt: torch.Tensor, **kwargs):

        raise NotImplementedError("prefix_tuning not implemented yet")

    def forward_features(self,
                         inputs: torch.Tensor,
                         text_tokens: torch.Tensor = None,
                         **kwargs) -> torch.Tensor:
        self.backbone.visual.eval()

        inputs = inputs.type(self.backbone.dtype)

        if text_tokens is None:
            text_tokens = self.text_tokens

        text_features = self.backbone.encode_text(text_tokens)

        # generate query
        with torch.no_grad():
            x = self.backbone.visual.conv1(inputs)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            B, N, D = x.size()

            x = torch.cat([
                self.backbone.visual.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                    device=x.device), x
            ],
                          dim=1)
            x = x + self.backbone.visual.positional_embedding.to(x.dtype)
            x = self.backbone.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            query = x.clone()
            for n, block in enumerate(
                    self.backbone.visual.transformer.resblocks):
                if n == len(self.backbone.visual.transformer.resblocks
                            ) - 1 and not self.use_last_layer:
                    break
                query = block(query)
            query = query.permute(1, 0, 2)
            query = self.backbone.visual.ln_post(query[:, 0, :])

        if self.training:
            self.features = torch.cat((self.features, query.detach().cpu()),
                                      dim=0)

        distance = 1 - F.cosine_similarity(
            query.unsqueeze(1), self.key, dim=-1)
        if self.use_contrastiv:
            mass = (self.count + 1)
        else:
            mass = 1.
        scaled_distance = distance * mass
        topk = scaled_distance.topk(self.selection_size, dim=1,
                                    largest=False)[1]
        distance = distance[torch.arange(topk.size(0), device=topk.device).
                            unsqueeze(1).repeat(1, self.selection_size),
                            topk].squeeze().clone()
        e_prompts = self.e_prompts[topk].squeeze().clone()
        mask = self.mask[topk].mean(1).squeeze().clone()

        if self.use_contrastiv:
            key_wise_distance = 1 - F.cosine_similarity(
                self.key.unsqueeze(1), self.key, dim=-1)
            self.similarity_loss = -(
                (key_wise_distance[topk] / mass[topk]).exp().mean() /
                ((distance / mass[topk]).exp().mean() +
                 (key_wise_distance[topk] / mass[topk]).exp().mean()) +
                1e-6).log()
        else:
            self.similarity_loss = distance.mean()

        g_prompts = self.g_prompts[0].repeat(B, 1, 1)
        if self.training:
            with torch.no_grad():
                num = topk.view(-1).bincount(minlength=self.e_prompts.size(0))
                self.count += num

        x = self.prompt_func(x, g_prompts.type(x.dtype),
                             e_prompts.type(x.dtype))
        x = x.permute(1, 0, 2)
        x = self.backbone.visual.ln_post(x[:, 0, :])
        if self.backbone.visual.proj is not None:
            x = x @ self.backbone.visual.proj

        mask = torch.sigmoid(mask) * 2.
        return x, text_features, mask[:, :self.text_tokens.shape[0]]

    def forward_head(self,
                     image_features: torch.Tensor,
                     text_features: torch.Tensor = None,
                     **kwargs) -> torch.Tensor:

        # normalized features
        image_features = image_features / image_features.norm(dim=1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.backbone.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        return logits_per_image

    def forward(self, inputs, text_tokens=None, **kwargs) -> torch.Tensor:
        x, text_features, mask = self.forward_features(inputs, text_tokens,
                                                       **kwargs)
        x = self.forward_head(x, text_features, **kwargs)
        if self.use_mask:
            x = x * mask
        return x

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target) + self.similarity_loss

    def get_similarity_loss(self):
        return self.similarity_loss

    def get_count(self):
        return self.prompt.update()
