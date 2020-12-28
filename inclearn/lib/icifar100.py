from torchvision.datasets import CIFAR100
import numpy as np
import torch
from PIL import Image

class iCIFAR100(CIFAR100):
    def __init__(self, root, classes=range(10), train=True, transform=None, 
                 target_transform=None, download=False):
      
        super(iCIFAR100, self).__init__(root, train=train, transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        # Select subset of classes
        data = []
        targets = []
        classes_new = []
        class_to_idx_new = {}

        for i in range(len(self.data)):
            if self.targets[i] in classes:
                data.append(self.data[i])
                targets.append(self.targets[i])
                if self.targets[i] not in classes_new:
                    for k, v in self.class_to_idx.items():
                        if v == self.targets[i]:
                            classes_new.append(self.targets[i])
                            class_to_idx_new[k] = v

        self.data = np.array(data)
        self.targets = targets
        self.classes = classes_new
        self.class_to_idx = class_to_idx_new

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]    

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        return len(self.data)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

    def append(self, images, labels):
        """Append dataset with images and labels
        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.data = np.concatenate((self.data, images), axis=0)
        self.targets = self.targets + labels