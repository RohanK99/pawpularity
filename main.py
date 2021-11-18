from data import *
from model import Model

import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning import callbacks
from tensorboard.backend.event_processing import event_accumulator

# dataloader = PawpularityDataModule().train_dataloader()
# images, labels = next(iter(dataloader))

# plt.figure(figsize=(12,12))
# for it, (image, label) in enumerate(zip(images, labels)):
#     plt.subplot(4,4,it+1)
#     plt.imshow(image.permute(1,2,0))
#     plt.axis('off')
#     plt.title(f'Pawpularity: {label}')

# plt.show()

datamodule = PawpularityDataModule()
model = Model()

logger = pl.loggers.TensorBoardLogger("tb_logs", "trash", default_hp_metric=False)
early_stopping_cb = callbacks.EarlyStopping(monitor="val_rmse")
model_cb = callbacks.ModelCheckpoint(
    filename='{epoch}-{val_loss:.2f}',
    monitor="val_rmse",
    save_top_k=1,
    mode="min",
    save_last=False,
)
trainer = pl.Trainer(
    logger=logger,
    max_epochs=2,
    callbacks=[early_stopping_cb, model_cb],
    gpus=1,
)
trainer.fit(model, datamodule=datamodule)

# print scalars when we need to calculate cv score
event_acc = event_accumulator.EventAccumulator('tb_logs/swinv1/version_1')
event_acc.Reload()

tag = "val_rmse"
for scalar_event in event_acc.Scalars(tag):
    print(scalar_event.value)