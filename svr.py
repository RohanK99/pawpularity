from data import *
from model import Model

from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold

import torch
import pytorch_lightning as pl
import torchmetrics
import numpy as np
import pickle

num_splits = 5
seed = 42
pl.utilities.seed.seed_everything(seed=seed)
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)

annotations = pd.read_csv('train.csv')
imgs = annotations["Id"].to_numpy()
labels = annotations["Pawpularity"].to_numpy()

model_path = 'trained-models/large-swin-224'
ensemble_rmse_sum = 0.0
for idx, (train_idx, val_idx) in enumerate(skf.split(imgs, labels)):
    datamodule = PawpularityDataModule(imgs[train_idx], labels[train_idx], imgs[val_idx], labels[val_idx])

    model = Model()
    model = Model.load_from_checkpoint(f"{model_path}/{idx}.ckpt")

    trainer = pl.Trainer(gpus=1)
    predictions = trainer.predict(model, datamodule.predict_dataloader())
    _, nn_features = zip(*predictions)

    # move preds back to cpu
    train_features = np.array([]).reshape((0,1536))
    for features in nn_features:
        train_features = np.concatenate((train_features, features.cpu().numpy()), axis=0)

    # train SVR head
    clf = SVR(C=20.0)
    clf.fit(train_features, labels[train_idx])

    # save clf
    pickle.dump(clf, open(f"{model_path}/svr/{idx}.pkl", "wb"))

    val_predictions = trainer.predict(model, datamodule.val_dataloader())
    nn_val_preds, nn_val_features = zip(*val_predictions)

    val_preds = np.array([]).reshape((0,1))
    for preds in nn_val_preds:
        val_preds = np.concatenate((val_preds, preds.cpu()), axis=0)
    val_preds = val_preds.flatten()
    val_preds = np.array([torch.sigmoid(torch.tensor(x)) * 100 for x in val_preds])
    
    val_features = np.array([]).reshape((0,1536))
    for features in nn_val_features:
        val_features = np.concatenate((val_features, features.cpu().numpy()), axis=0)

    clf_preds = clf.predict(val_features)
    ensemble_preds = 0.8*val_preds + 0.2*clf_preds
    
    nn_rmse = torchmetrics.functional.mean_squared_error(torch.tensor(val_preds), torch.tensor(labels[val_idx]), squared=False)
    clf_rmse = torchmetrics.functional.mean_squared_error(torch.tensor(clf_preds), torch.tensor(labels[val_idx]), squared=False)
    ensemble_rmse = torchmetrics.functional.mean_squared_error(torch.tensor(ensemble_preds), torch.tensor(labels[val_idx]), squared=False)
    ensemble_rmse_sum += ensemble_rmse
    print(f"NN RMSE: {nn_rmse}")
    print(f"SVR RMSE: {clf_rmse}")
    print(f"Ensemble RMSE: {ensemble_rmse}")

ensemble_cv_rmse = ensemble_rmse_sum / num_splits
print(f"Ensemble CV Score: {ensemble_cv_rmse}")