from arch import *


simclr_path = "/Users/allen/扣得/l3_dectector/saved_models/SimCLR/lightning_logs/version_29/checkpoints/epoch=4-step=1640.ckpt"
linear_path = '/Users/allen/扣得/l3_dectector/saved_models/LogisticRegression/lightning_logs/version_14/checkpoints/epoch=59-step=4200.ckpt'

class simclr(nn.Module):
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
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)
        self.sigmoid = nn.Sigmoid()  # 添加 Sigmoid 激活函數

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)  # 將線性層的輸出通過 Sigmoid 函數
        return x

    
class CombinedModel(nn.Module):
    def __init__(self, simclr, linear):
        super(CombinedModel, self).__init__()
        self.part1 = simclr
        self.part2 = linear
        
    def forward(self, x):
        # Forward pass through the first part of the model
        x_part1 = self.part1(x)
        
        # Forward pass through the second part of the model
        x_part2 = self.part2(x_part1)
        
        return x_part2

simclr_model = simclr(128)
simclr_model.load_state_dict(torch.load(simclr_path)['state_dict'])

linear_model = linear(feature_dim=128, num_classes=12)
linear_model.load_state_dict(torch.load(linear_path)['state_dict'])

complete_model = CombinedModel(simclr=simclr_model, linear=linear_model)

torch.save(complete_model, '/Users/allen/扣得/l3_dectector/models/complete_model.pth')