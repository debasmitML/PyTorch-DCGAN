import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ganDataset(Dataset):
    
    
    def __init__(self , data_dir = 'data/img_align_celeba'):
        
        self.data_list = glob.glob(os.path.join(data_dir,'*'))
        self.transform = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.Resize((64,64),antialias=True),
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
            
        ])
           
    def __len__(self):
        
        return len(self.data_list)
        
    def __getitem__(self, index):
        
        img_arr = cv2.imread(self.data_list[index])
        rgb_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        normalized_img_tensor = self.transform(rgb_img)
        
        return normalized_img_tensor
    
