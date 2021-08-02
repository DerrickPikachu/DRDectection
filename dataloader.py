import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from skimage import io
import os

from imgTransform import ImgToTorch


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, transform=None):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform = transform
        # print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        imagePath = os.path.join(self.root, self.img_name[index] + '.jpeg')
        img = io.imread(imagePath)
        label = self.label[index]

        if self.transform:
            img = self.transform(img)

        return img, label


class PretrainLoader(RetinopathyLoader):
    def __init__(self, root, mode, transform=None):
        super(PretrainLoader, self).__init__(root, mode, transform)

        # Leave data that is not belong to class 0
        no_class0_img = []
        no_class0_label = []
        for i in range(len(self.img_name)):
            if self.label[i] != 0:
                no_class0_img.append(self.img_name[i])
                no_class0_label.append(self.label[i])

        self.img_name = np.asarray(no_class0_img)
        self.label = np.asarray(no_class0_label)

        # print("> Found %d images..." % (len(self.img_name)))


class EvenLoader(RetinopathyLoader):
    def __init__(self, root, mode, transform=None):
        super(EvenLoader, self).__init__(root, mode, transform)

        counter = 3000
        even_img = []
        even_label = []
        for i in range(len(self.img_name)):
            if self.label[i] != 0 or (self.label[i] == 0 and counter != 0):
                even_img.append(self.img_name[i])
                even_label.append(self.label[i])
                if self.label[i] == 0:
                    counter -= 1

        self.img_name = np.asarray(even_img)
        self.label = np.asarray(even_label)

        print("> EvenLoader Found %d images..." % (len(self.img_name)))


if __name__ == "__main__":
    # retinopathyData = RetinopathyLoader('data', 'train', transform=ImgToTorch())
    # img, label = retinopathyData[0]
    # plt.figure()
    # plt.imshow(img.numpy().transpose((1, 2, 0)))
    # plt.show()

    # no_class0_data = PretrainLoader('data', 'train', transform=ImgToTorch())
    # print(no_class0_data.label)

    img_name, label = getData('test')
    total_size = len(label)
    counter = np.zeros(5).astype('int')

    for num in label:
        counter[num] += 1

    counter = counter.astype('float') / total_size
    for i in range(len(counter)):
        print(f'class {i}: {counter[i] * 100:.4f}')

