import torch.nn as nn
from transformers import AutoModel

from gryph.utils import MODELS


@MODELS.register_module()
class FormulaEncoder(nn.Module):
    def __init__(self, backbone: str, **kwargs):
        super().__init__()

        # backbone
        self.backbone = AutoModel.from_pretrained(backbone, **kwargs)

    def forward(self, img, train: bool, **kwargs):
        return dict(kwargs, img=self.backbone(img).last_hidden_state)
