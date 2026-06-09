import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.nn import BatchNorm
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

NODE_FEATS = 4
GNN_HIDDEN = 128

class DyslexiaGCN(nn.Module):
    """
    3-layer GCN → global mean pool → 2-layer MLP → class logits
    """
    def __init__(self, in_channels=NODE_FEATS, hidden=GNN_HIDDEN, n_classes=2):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for DyslexiaGCN")
            
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1 = BatchNorm(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = BatchNorm(hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn3 = BatchNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), 
            nn.ReLU(), 
            nn.Dropout(0.4), 
            nn.Linear(hidden // 2, n_classes)
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.head(x)

class SimpleCNN(nn.Module):
    """
    3 convolutional blocks trained from scratch on scanpath images.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(0.5), 
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_resnet18(num_classes=2):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), 
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def get_efficientnet_b0(num_classes=2):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model
