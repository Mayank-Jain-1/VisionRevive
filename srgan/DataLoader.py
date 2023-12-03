import os
import cv2
import numpy as np
import DataCreator
import torch
from torch.utils.data import Dataset, DataLoader

class MyImageFolder(Dataset):
    def __init__(self, hr_folder, lr_folder):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = hr_folder
        self.images_names = os.listdir(hr_folder) 

        for name in self.images_names:
            hr_path = os.path.join(hr_folder, name)
            lr_path = os.path.join(lr_folder, name)
            self.data.append((hr_path, lr_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        hr_img = cv2.imread(self.data[index][0])
        lr_img = cv2.imread(self.data[index][1])
        hr_img = torch.tensor(hr_img).type(torch.float32).permute((2,0,1))/255
        lr_img = torch.tensor(lr_img).type(torch.float32).permute((2,0,1))/255
        return hr_img, lr_img
    

def test():
    hr_folder = ".\Dataset\DIV2K_train_hr"
    lr_folder = ".\Dataset\DIV2K_train_lr"
    loader = MyImageFolder(hr_folder, lr_folder)
    for i in range(5):
        high_res, low_res = loader.__getitem__(i)
        cv2.imwrite(f"./Dataset/TrainingSamples/{i}testhr.png", high_res)
        cv2.imwrite(f"./Dataset/TrainingSamples/{i}testlr.png", low_res)

if __name__ == "__main__":
    test()