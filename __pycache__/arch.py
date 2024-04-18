import torch
import torch.nn as nn
import torchvision.models as models


class simclr(nn.Module): # type: ignore
    def __init__(self, hidden_dim):
        super(simclr, self).__init__()
        
        # 創建一個ResNet-34模型
        self.convnet = models.resnet34(num_classes=4*hidden_dim)
        
        # 修改最後一層全連接層
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.convnet(x)
    
class linear(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        # Mapping from representation h to classesx
        self.model = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        return self.convnet(x)