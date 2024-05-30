# The codes in this directory were from https://github.com/drimpossible/GDumb/tree/master/src/models

from .dualprompt import DualPrompt
from .l2p import L2P
from .mvp import MVP
from .continual_clip import ContinualCLIP
from .mvp_clip import CLIP_MVP
from .maple import MaPLe
from .adapter_clip import AdapterCLIP


def get_model(method, model_name, **kwargs):
    if method == "dualprompt":
        return DualPrompt(**kwargs), 224
    elif method == "l2p":
        return L2P(**kwargs), 224
    elif method == "mvp":
        return MVP(**kwargs), 224
    elif method == "continual-clip":
        return ContinualCLIP(model_name=model_name,
                             device=kwargs['device']), 224
    elif method == "adapter-clip":
        return AdapterCLIP(model_name=model_name,
                           device=kwargs['device'],
                           peft_method='adapter',
                           peft_encoder=kwargs['peft_encoder']), 224
    elif method == "lora-clip":
        return AdapterCLIP(model_name=model_name,
                           device=kwargs['device'],
                           peft_method='lora',
                           peft_encoder=kwargs['peft_encoder']), 224
    elif method == "mvp-clip":
        return CLIP_MVP(model_name=model_name, **kwargs), 224
    elif method == "maple":
        return MaPLe(model_name=model_name, n_ctx=3,
                     device=kwargs['device']), 224
    else:
        raise NotImplementedError(
            f"Model {method}_{model_name} not implemented")
