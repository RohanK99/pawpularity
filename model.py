from pytorch_lightning import LightningModule

import torch
import torchmetrics
import torch.nn as nn
import timm

class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=0)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.LazyLinear(1) # we only want one output feature and then perform sigmoid
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch, 'train')
        self.train_rmse(preds, labels)
        self.log("train_rmse", self.train_rmse, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch, 'val')
        self.val_rmse(preds, labels)
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)
    
    def __share_step(self, batch, mode):
        images, labels = batch
        logits = self.forward(images).squeeze(1)
        loss = self.criterion(logits, labels) # BCE loss already includes sigmoid
        
        preds = torch.sigmoid(logits).detach().cpu() * 100
        labels = labels.detach().cpu() * 100
        return loss, preds, labels

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = 20,
            eta_min = 1e-4
        )
        return [optimizer], [scheduler]