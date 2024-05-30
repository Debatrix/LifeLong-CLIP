import torch
import torch.nn as nn

import clip


class ContinualCLIP(nn.Module):

    def __init__(self, model_name='ViT-B/16', device='cpu'):
        super(ContinualCLIP, self).__init__()
        self.device = device

        self.model, _ = clip.load(model_name, device=self.device, jit=False)

        self.text_tokens = None
        self.current_class_names = []

        self.prompt_template = "a bad photo of a {}."

    def update_class_names(self, new_class_names):
        _num = 0
        for c in new_class_names:
            if c not in self.current_class_names:
                self.current_class_names.append(c)
                _num += 1
        if _num > 0:
            self.text_tokens = clip.tokenize([
                self.prompt_template.format(c)
                for c in self.current_class_names
            ])
        self.text_tokens = self.text_tokens.to(self.device)
        return self.text_tokens

    def forward(self, image):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image.softmax(dim=-1)
        return probs
