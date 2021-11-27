from data import *
from model import Model

import glob
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning import callbacks
from tensorboard.backend.event_processing import event_accumulator
from sklearn.model_selection import StratifiedKFold

# dataloader = PawpularityDataModule().train_dataloader()
# images, labels = next(iter(dataloader))

# plt.figure(figsize=(12,12))
# for it, (image, label) in enumerate(zip(images, labels)):
#     plt.subplot(4,4,it+1)
#     plt.imshow(image.permute(1,2,0))
#     plt.axis('off')
#     plt.title(f'Pawpularity: {label}')

# plt.show()

# config options
log_folder = "tb_logs"
experiment_name="kfold-large-swinv1"
version_num = "version_0"
num_splits = 5
seed = 42

pl.utilities.seed.seed_everything(seed=seed)

# read data
annotations = pd.read_csv('train.csv')
imgs = annotations["Id"].to_numpy()
labels = annotations["Pawpularity"].to_numpy()

# train model
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
logger = pl.loggers.TensorBoardLogger(log_folder, experiment_name, default_hp_metric=False)
for train_idx, val_idx in skf.split(imgs, labels):
    datamodule = PawpularityDataModule(imgs[train_idx], labels[train_idx], imgs[val_idx], labels[val_idx])
    model = Model()

    early_stopping_cb = callbacks.EarlyStopping(monitor="val_rmse")
    model_cb = callbacks.ModelCheckpoint(
        filename='{epoch}-{val_rmse:.2f}',
        monitor="val_rmse",
        save_top_k=1,
        mode="min",
        save_last=False,
    )
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=20,
        callbacks=[early_stopping_cb, model_cb],
        gpus=1,
    )
    trainer.fit(model, datamodule=datamodule)

# calculate cross validation score
rmse_sum = 0.0
for event_file in glob.glob(f'{log_folder}/{experiment_name}/{version_num}/events.*'):
    event_acc = event_accumulator.EventAccumulator(event_file)
    event_acc.Reload()

    tag = "val_rmse"
    rmse = []
    for scalar_event in event_acc.Scalars(tag):
        rmse.append(scalar_event.value)

    print(f"Event file {event_file} has min rmse of {min(rmse)}")
    rmse_sum += min(rmse)

cv_score = rmse_sum / num_splits
print(f'Cross Validation RMSE: {cv_score}')