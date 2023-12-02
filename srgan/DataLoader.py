import os
import cv2
import numpy as np
import DataCreator
from torch.utils.data import Dataset, DataLoader

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.images_names = os.listdir(root_dir) 

        for name in self.images_names:
            files = os.path.join(root_dir, name)
            self.data.append(files)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = np.array(cv2.imread(self.data[index]))
        low_res = DataCreator.createLrImage(image=image, mosaic_intensity=5, noise_intensity=0.35, cloudiness=0.4)
        high_res = DataCreator.center_crop(image=image)
        return high_res, low_res
    

def test():
    folder = ".\Dataset\DIV2K_train_HR"
    loader = MyImageFolder(folder)
    for i in range(5):
        high_res, low_res = loader.__getitem__(i)
        cv2.imwrite(f"./Dataset/TrainingSamples/{i}testhr.png", high_res)
        cv2.imwrite(f"./Dataset/TrainingSamples/{i}testlr.png", low_res)

if __name__ == "__main__":
    test()