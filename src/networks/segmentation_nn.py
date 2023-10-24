import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models.segmentation.deeplabv3 

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1, targets.size(-1))

        intersection = (inputs * targets).sum(dim=0)
        dice_coefficient = (2. * intersection + smooth) / (inputs.sum(dim=0) + targets.sum(dim=0) + smooth)

        dice_loss = 1 - dice_coefficient.mean()

        return dice_loss

class SegmentationNN(pl.LightningModule):
    
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        model.backbone.layer4[0].conv2.stride = (1, 1)
        model.backbone.layer4[0].downsample[0].stride = (1, 1)

        model.classifier[-1] = torch.nn.Conv2d(256, self.hparams['num_classes'], kernel_size=(1, 1), stride=(1, 1))

        self.model=model
        self.act = nn.Softmax(dim=1)  

    def forward(self, x):

        x=self.act(self.model(x)['out'])
        out=self.act(x)

        return out

    def training_step(self, batch, batch_idx):
        images, targets = batch

        out = self.forward(images)

        loss_func = DiceLoss()
        loss=loss_func(out, targets)
        self.log('loss', loss, logger=True, prog_bar=True)
      
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        out = self.forward(images)

        loss_func = DiceLoss()
        loss=loss_func(out, targets)*100

        self.log("val_loss", loss, logger=True, prog_bar=True)

        return {'val_loss': loss}

    def configure_optimizers(self):

        optim=torch.optim.Adam(self.parameters(),lr=self.hparams['lr'])        
        return optim