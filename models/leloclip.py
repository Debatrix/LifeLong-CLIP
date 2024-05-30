import copy
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.maple_clip import clip
from models.maple_clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

logger = logging.getLogger()
_tokenizer = _Tokenizer()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def load_clip(backbone_name, n_ctx=2, device='cpu'):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {
        "trainer": 'MaPLe',
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": n_ctx
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model.to(device)


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts,
                compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [
            x, compound_prompts_deeper_text, 0
        ]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptPoolLearner(nn.Module):

    def __init__(self, clip_model, n_ctx=2):
        super().__init__()
        self.current_class_names = []
        ctx_init = "a bad photo of a"
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Default is 1, which is compound shallow prompting
        # For MaPLe, PROMPT_DEPTH should be >= 1
        # self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        self.compound_prompts_depth = 3

        self.n_ctx = n_ctx
        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            device = clip_model.ln_final.weight.device
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            self.prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768

        self.proj = nn.Linear(ctx_dim, 768).to(self.dtype)
        # self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx, 512))
            for _ in range(self.compound_prompts_depth - 1)
        ])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, prefix, suffix):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(prefix.shape[0], -1, -1)

        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(
                self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(
            self.ctx
        ), self.compound_prompts_text, visual_deep_prompts  # pass here original, as for visual 768 is required


class LeLoClip(nn.Module):

    def __init__(self, model_name="ViT-B/16", n_ctx=3, device='cpu'):
        super().__init__()
        self.device = device

        clip_model = load_clip(model_name, n_ctx=n_ctx)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_ctx = n_ctx

        self.prompt_learner = MultiModalPromptLearner(clip_model, n_ctx=n_ctx)

        self.register_buffer("token_prefix", torch.zeros(0))  # SOS
        self.register_buffer("token_suffix", torch.zeros(0))  # CLS, EOS
        self.tokenized_prompts = None

        self.current_class_names = []

        self.prompt_prefix = self.prompt_learner.prompt_prefix

    def update_class_names(self, new_class_names):
        _num = 0
        for c in new_class_names:
            if c not in self.current_class_names:
                self.current_class_names.append(c)
                _num += 1
        if _num > 0:
            self.tokenized_prompts, self.token_prefix, self.token_suffix = self.get_tokenized_prompts(
                self.current_class_names)
        return self.tokenized_prompts

    def get_tokenized_prompts(self, classnames):
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [
            self.prompt_prefix + " " + name + "." for name in classnames
        ]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts
                                       ]).to(self.device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = self.text_encoder.token_embedding(
                tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        token_prefix = embedding[:, :1, :]  # SOS
        token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS

        return tokenized_prompts, token_prefix, token_suffix  # torch.Tensor

    def forward(self, image, tokenized_prompts=None, prefix=None, suffix=None):

        logit_scale = self.logit_scale.exp()

        if tokenized_prompts is None:
            tokenized_prompts = self.tokenized_prompts
            prefix = self.token_prefix
            suffix = self.token_suffix

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(
            prefix, suffix)

        text_features = self.text_encoder(prompts, tokenized_prompts,
                                          deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx,
                                            deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,
                                                           keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        # probs = logits.softmax(dim=-1)
        return logits
