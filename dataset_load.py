from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as transforms
import torchvision.transforms as T
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np


class LoadDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.files = glob.glob(file_list + '/*.png')  # church
        '''
        self.folders = glob.glob(file_list+ '/*')
        self.files=[]
        for folder in self.folders:
            for f in glob.glob(folder+'/*.png'):
                self.files.append(f)
        '''

        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = cv2.imread(self.files[index], 3)
        img = cv2.resize(img, (256, 256))
        im = img[:, :, :3] / 255
        im = np.stack((im[:, :, 2], im[:, :, 1], im[:, :, 0]), axis=2)

        img_all = np.stack((im[:, :, 2], im[:, :, 1], im[:, :, 0]), axis=2)
        return img_all.transpose(2, 0, 1)