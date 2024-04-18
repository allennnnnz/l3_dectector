from combine_model import *
from PIL import Image
import torchvision.transforms as transforms


class_names = ['L0', 'L1', 'L1_L2', 'L2', 'L2_L3', 'L3', 'L3_L4', 'L4', 'L4_L5', 'L5', 'T12', 'T12_L1']

# 以 PIL 影像格式讀取圖片
img = Image.open('/Users/allen/扣得/l3_dectector/data/test/L0/s0032_56_T10_T11.png')
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 調整大小為 224x224
    transforms.ToTensor(),  # 轉換為 PyTorch 張量
    transforms.Normalize(mean=0.187, std=0.2518582458582089)  # 正規化
])

#numpy算圖片 RGB 的mean std
if img.mode != 'RGB':
    img = img.convert('RGB')
    
# 將影像應用轉換
img = transform(img)

# 添加一個批次的維度 (因為模型預期的輸入是批次的)
img = img.unsqueeze(0)

complete_model.eval()



with torch.no_grad():
    output = complete_model(img)
    max_index = torch.argmax(output)
    print(output)
    print("Predicted class:", class_names[max_index.item()])