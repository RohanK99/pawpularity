from data import *
from model import Model

import pytorch_lightning as pl
import matplotlib.pyplot as plt

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

logger = pl.loggers.TensorBoardLogger("tb_logs", "swinv1", default_hp_metric=False)
trainer = pl.Trainer(
    logger=logger,
    max_epochs=20,
    gpus=1,
)
trainer.fit(model, datamodule=datamodule)