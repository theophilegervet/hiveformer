# Adapted from https://github.com/openai/CLIP/blob/main/clip/model.py

import torch

import clip
# from clip.model import ModifiedResNet


def load_clip():
    clip_model, clip_transforms = clip.load("RN50")
    backbone = ResNet(clip_model.visual)
    # state_dict = clip_model.state_dict()
    # layers = tuple([len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
                    # for b in [1, 2, 3, 4]])
    # output_dim = state_dict["text_projection"].shape[1]
    # heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64
    # backbone = ModifiedResNetFeatures(layers, output_dim, heads)
    # backbone.load_state_dict(clip_model.visual.state_dict())
    normalize = clip_transforms.transforms[-1]
    return backbone, normalize

class ResNet(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, x):
        x = x.type(self.visual.conv1.weight.dtype)
        x = self.visual.relu1(self.visual.bn1(self.visual.conv1(x)))
        x = self.visual.relu2(self.visual.bn2(self.visual.conv2(x)))
        x0 = self.visual.relu3(self.visual.bn3(self.visual.conv3(x)))
        x = self.visual.avgpool(x0)
        x1 = self.visual.layer1(x)
        x2 = self.visual.layer2(x1)
        x3 = self.visual.layer3(x2)
        x4 = self.visual.layer4(x3)

        return {
            "res1": x0.float(),
            "res2": x1.float(),
            "res3": x2.float(),
            "res4": x3.float(),
            "res5": x4.float(),
        }

