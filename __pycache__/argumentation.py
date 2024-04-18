import torchvision
from torchvision import transforms

class ContrastiveTransformations(object):
    
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
        
    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    
    
contrast_transforms = transforms.Compose([
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.7, 
                                                                     contrast=0.7, 
                                                                     saturation=0.5, 
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                         ])
#隨機調整亮度、對比度、飽和度和色調，隨機轉換為灰度圖像，以及應用高斯模糊