import torch
import torch.nn as nn
from torchvision import models

def load_densenet(device):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Identity()
    model.eval()
    model.to(device)
    return model
