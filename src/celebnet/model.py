import torch, torch.nn as nn
from facenet_pytorch import InceptionResnetV1

def build_backbone(device):
    net = InceptionResnetV1(pretrained='vggface2', classify=False).to(device).eval()
    for p in net.parameters(): p.requires_grad=False
    return net

def build_head(num_classes, device):
    return nn.Linear(512, num_classes).to(device)
