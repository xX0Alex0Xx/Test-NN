import numpy as np
import torch.utils.data as data
from torchvision import transforms
import cv2

class SegmentationData(data.Dataset):

    def __init__(self,hparams, images_path,labels_path,test):
        self.images_path = images_path
        self.labels_path = labels_path
        self.test=test
        self.hparams=hparams
        self.transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.transform_label = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, key):
            if isinstance(key, slice):
                return [self[ii] for ii in range(*key.indices(len(self)))]
            elif isinstance(key, int):
                if key < 0:
                    key += len(self)
                if key < 0 or key >= len(self):
                    raise IndexError("The index (%d) is out of range." % key)
                return self.get_item_from_index(key)
            else:
                raise TypeError("Invalid argument type.")

    def get_item_from_index(self, index):
        def encode_labels(mask):
            label_mask = np.zeros([mask.shape[0],mask.shape[1],self.hparams['num_classes']])
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    label_mask[i,j,self.hparams['color_coder'][mask[i,j,1]]] = 1
            return label_mask
            
        img = cv2.imread(self.images_path[index])
        lbl = cv2.imread(self.labels_path[index])
        label=encode_labels(lbl)

        if self.test==False:
            cropped_image, cropped_label = self.crop_image(img, label)
            image = self.transform_img(cropped_image)   
            label = self.transform_label(cropped_label) 
        else:
            image = self.transform_img(img)   
            label = self.transform_label(label) 

        return image,label
    
    def crop_image(self, image, label):
        h, w = image.shape[:2]
        x = np.random.randint(0, w - self.hparams["crop_size"])
        y = np.random.randint(0, h - self.hparams["crop_size"])
        cropped_image = image[y:y+self.hparams["crop_size"], x:x+self.hparams["crop_size"]]
        cropped_label = label[y:y+self.hparams["crop_size"], x:x+self.hparams["crop_size"]]
        return cropped_image, cropped_label