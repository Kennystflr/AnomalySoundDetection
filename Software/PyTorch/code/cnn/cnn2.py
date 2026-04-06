import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtBinary(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained ConvNeXt (tiny is a good starting point)
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

        # 2. Freeze the backbone - the backbone doesn't train anymore (it's already pre-trained)
        #for param in self.model.features.parameters():
            #param.requires_grad = False

        # Get input features of classifier
        in_features = self.model.classifier[2].in_features

        # Replace classifier head for binary classification
        self.model.classifier[2] = nn.Linear(in_features, 1)


    def forward(self, x):
        return self.model(x)


