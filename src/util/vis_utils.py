import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

def accuracy(model,test_data,clases):

    def calculate_iou(predicted, target, clases):
        iou_per_class = []
        eps = 1e-6  

        for class_idx in range(clases):
            intersection = torch.logical_and(predicted == class_idx, target == class_idx).sum()
            union = torch.logical_or(predicted == class_idx, target == class_idx).sum()
            iou = (intersection + eps) / (union + eps)
            iou_per_class.append(iou.item())

        return iou_per_class
    model.eval()

    iou=np.zeros(clases)

    for i, (img, target) in enumerate(test_data):
        inputs = img.unsqueeze(0)
        outputs = model.forward(inputs)
        
        pred = outputs[0].data.cpu()
        target, pred = target.numpy(), pred.numpy()

        pred1=torch.from_numpy(np.argmax(pred,axis=0))
        target1=torch.from_numpy(np.argmax(target,axis=0))

        iou=iou+calculate_iou(pred1,target1,clases)

    iou_mean=np.sum(iou/(i+1))/clases
    return iou_mean*100, iou/(i+1)*100

def decode(enc,color_map):
    color_mask = np.zeros([enc.shape[1], enc.shape[2],3])
    enc1=np.argmax(enc,axis=0)
    for i in range(0,9):
        enc2= (enc1==i)
        color=color_map[i]
        color_mask[enc2]=color
    return color_mask/255

def visualizer(model,num_example_imgs=1, test_data=None,color_map=None):
    model.eval()
    plt.figure(figsize=(15, 5 * num_example_imgs))
    for i, (img, target) in enumerate(test_data[:num_example_imgs]):
        inputs = img.unsqueeze(0)
        outputs = model.forward(inputs)
        pred = outputs[0].data.cpu()
        
        img, target, pred = img.numpy(), target.numpy(), pred.numpy()

        pred1=decode(pred,color_map)
        target1=decode(target,color_map)
        # img
        plt.subplot(num_example_imgs, 3, i * 3 + 1)
        plt.axis('off')
        plt.imshow(img.transpose(1, 2, 0))
        if i == 0:
            plt.title("Input image")

        # target
        plt.subplot(num_example_imgs, 3, i * 3 + 2)
        plt.axis('off')
        plt.imshow(target1.transpose(0, 1, 2))
        if i == 0:
            plt.title("Target image")

        # pred

        plt.subplot(num_example_imgs, 3, i * 3 + 3)
        plt.axis('off')
        plt.imshow(pred1.transpose(0, 1, 2))
        if i == 0:
            plt.title("Prediction image")

    plt.show()