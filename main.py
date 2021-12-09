from data import *
from model import Model

import glob
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning import callbacks
from tensorboard.backend.event_processing import event_accumulator
from sklearn.model_selection import StratifiedKFold

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

def visualize_data():
    dataloader = PawpularityDataModule().train_dataloader()
    images, labels = next(iter(dataloader))

    plt.figure(figsize=(12,12))
    for it, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(4,4,it+1)
        plt.imshow(image.permute(1,2,0))
        plt.axis('off')
        plt.title(f'Pawpularity: {label}')

    plt.show()

# tune model hyperparameters
def train_tune(config):
    annotations = pd.read_csv('/home/docker/pawpularity/train.csv')
    imgs = annotations["Id"].to_numpy()
    labels = annotations["Pawpularity"].to_numpy()

    # read data
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    logger = pl.loggers.TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".")
    for fold, (train_idx, val_idx) in enumerate(skf.split(imgs, labels)):
        if (fold != 0):
            break
        
        datamodule = PawpularityDataModule(imgs[train_idx], labels[train_idx], imgs[val_idx], labels[val_idx], config)
        model = Model(config)

        early_stopping_cb = callbacks.EarlyStopping(monitor="val_rmse")
        tune_cb = TuneReportCallback({"rmse": "val_rmse"}, on="validation_end")
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=20,
            callbacks=[tune_cb, early_stopping_cb],
            gpus=1,
            progress_bar_refresh_rate=0
        )
        trainer.fit(model, datamodule=datamodule)

num_splits = 5
seed = 42

pl.utilities.seed.seed_everything(seed=seed)

config = {
    "batch_size": tune.choice([8, 16, 32]),
    "lr": tune.loguniform(1e-6, 1e-2),
    "T_0": tune.choice([10, 20, 50, 100]),
    "eta_min": tune.loguniform(1e-6, 1e-2)
}

scheduler = ASHAScheduler(
    max_t=20,
    grace_period=3,
    reduction_factor=2)

reporter = CLIReporter(
    parameter_columns=["batch_size", "lr", "T_0", "eta_min"],
    metric_columns=["rmse", "training_iteration"])

analysis = tune.run(train_tune,
    resources_per_trial = {"cpu": 12, "gpu": 1},
    metric="rmse",
    mode="min",
    config=config,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter,
    name="tune_asha")

print("Best hyperparameters found were: ", analysis.best_config)