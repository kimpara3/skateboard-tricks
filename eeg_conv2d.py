import math
import pickle
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import torchmetrics
from torchinfo import summary
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import mlflow

from eeg_utils import *

os.environ["no_proxy"]="localhost, 127.0.0.1, ::1"
mlflow_tracking_uri="http://localhost:5000"

class EEG2dDataset(Dataset):
    def __init__(self, signals, labels=None, transform=None):
        '''
        signals : (N,T,C)
        labels : (N,)
        output : (N,C1,C2,T)
        '''
        ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'FCz', 'AFz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Fpz', 'CPz', 'POz', 'Iz', 'F9', 'F10', 'P9', 'P10', 'PO9', 'PO10', 'O9', 'O10']
        ch_groups = [
            "F10 F8 FC6 C6 CP6 P8 P10 FT10 FT8 T8 TP8".split(),
            "AF8 F4 F6 FC4 C4 CP4 P4 P6 PO8 PO10".split(),
            "Fp2 AF4 F2 FC2 C2 CP2 P2 PO4 O2 O10".split(),
            "Fpz AFz Fz FCz Cz CPz Pz POz Oz Iz".split(),
            "Fp1 AF3 F1 FC1 C1 CP1 P1 PO3 O1 O9".split(),
            "AF7 F3 F5 FC3 C3 CP3 P3 P5 PO7 PO9".split(),
            "F9 F7 FC5 C5 CP5 P7 P9 FT9 FT7 T7 TP7".split(),
        ]
        N,T,C = signals.shape
        C1 = max([len(se) for se in ch_groups])
        C2 = len(ch_groups)
        topo_like = np.zeros((N, C1, C2, T),dtype=np.float32)

        for i2, group in enumerate(ch_groups):
            for i1, channel in enumerate(group):
                ch_index = ch_names.index(channel)
                topo_like[:, i1, i2, :] = signals[:, :, ch_index]

        self.topo_like = torch.tensor(topo_like, dtype=torch.float32)
        self.labels = torch.tensor(labels % 10 - 1) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.topo_like)

    def __getitem__(self, idx):
        T = self.topo_like.size(3)
        if T > 250:
            start_idx = np.random.randint(T - 250 + 1)
            end_idx = start_idx + 250
            ret_X = self.topo_like[idx,:,:,start_idx:end_idx]
        else:
            ret_X = self.topo_like[idx]

        if self.labels is not None:
            # train
            return ret_X, self.labels[idx]
        else:
            # predict
            return ret_X


class EEG2dClassificationModel(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-5, weight_decay=0):
        super(EEG2dClassificationModel, self).__init__()
        self.save_hyperparameters()

        self.model = self._create_model(11)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

    def _create_model(self, in_channles):
        layers = [
            nn.Conv2d(in_channels=in_channles, out_channels=128, kernel_size=(3,7), stride=(1,3), padding=(1,3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,7), stride=(1,3), padding=(1,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((1, 1)),  # AdaptiveAvgPool2d added before flatten
            nn.Flatten(),
            nn.Linear(64, self.hparams.num_classes)  # Final FC layer
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        self.val_acc(predicted, labels)

        self.log("val_loss", loss)
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, reduce_fx="mean")
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer
    
def normalize_trial(X):
    std=X.std(axis=(0,1),keepdims=True)
    std[std==0.0]=1.0
    X = X/(2*std)
    X = np.clip(X, -6,6)
    return X

# データの準備
train_data, target = prepare_all() # (N,T,C)

for sub in "0000 0001 0002 0003 0004 ".split():
    kf=KFold(3)
    for isplit, (train_index, val_index) in enumerate(kf.split(train_data[sub])):
        seed_everything(1, workers=True)

        normalized_data = []
        for X in [train_data[sub][i] for i in range(3)]:
            X = normalize_trial(X)
            normalized_data.append(X)

        train_X = np.concatenate([normalized_data[i] for i in train_index], axis=0)
        val_X = normalized_data[val_index[0]][:, 25:25+250]

        train_y = np.concatenate([target[sub][i] for i in train_index], axis=0)
        val_y = target[sub][val_index[0]]

        train_dataset = EEG2dDataset(train_X, train_y)
        val_dataset = EEG2dDataset(val_X, val_y)

        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=math.ceil(len(val_dataset)/2), shuffle=False, num_workers=1, pin_memory=True)

        model = EEG2dClassificationModel(num_classes=3, learning_rate=2e-4, weight_decay=1e-4)

        print(train_dataset[:1][0].shape)
        model_summary=summary(model=model.model, input_size=train_dataset[:1][0].shape)

        # MLflow Loggerの設定
        cond="final"
        mlflow_logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name="conv2d2",
            run_name=f"{cond} {sub}-{isplit}",
            log_model = True,
        )
        mlflow_logger.log_hyperparams({
            "condition":cond,
            "sub":sub,
            "isplit":isplit,
            "batch_size":train_dataloader.batch_size,
        })
        mlflow_logger.experiment.log_text(mlflow_logger.run_id, str(model_summary), "model_summary.txt")
        mlflow_logger.experiment.log_text(mlflow_logger.run_id, str(model.model), "model.txt")

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=None,
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )
        trainer = Trainer(
            max_epochs=200,
            accelerator='gpu',
            callbacks=[checkpoint_callback],
            deterministic=True,
            logger=mlflow_logger,
            log_every_n_steps=len(train_dataloader),
            default_root_dir=None,
            enable_progress_bar=False,
        )
        # 学習
        trainer.fit(model, train_dataloader, val_dataloader)

        # checkpointのロード
        local_path = mlflow.artifacts.download_artifacts(
            run_id=mlflow_logger.run_id,
            artifact_path="model/checkpoints/best-checkpoint/best-checkpoint.ckpt"
        )
        model = EEG2dClassificationModel.load_from_checkpoint(local_path)
        model.eval()
        model=model.cuda()

        # validation score
        with torch.no_grad():
            all_predictions = []
            for x,label in val_dataloader:
                preds = model(x.cuda())
                all_predictions.append(preds)
            predictions = torch.cat(all_predictions, dim=0)
            predictions_dict = {f"{sub}-{isplit}":predictions.tolist()}
            pred_y = torch.argmax(predictions, 1)+1

        pred_y = pred_y.cpu().numpy()
        cm = confusion_matrix(val_y-10, pred_y)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig = disp.plot()
        mlflow_logger.experiment.log_figure(mlflow_logger.run_id, fig.figure_, f"confusion_matrix_class.png")
        mlflow_logger.experiment.log_metric(mlflow_logger.run_id, "accuracy", np.sum(pred_y==val_y%10)/len(val_y))
        mlflow_logger.experiment.log_dict(
            mlflow_logger.run_id, 
            dictionary=predictions_dict,
            artifact_file="val_predictions.json"
        )

        # 推論
        test_data = prepare_testdata()
        test_X = test_data[sub]
        test_X = normalize_trial(test_X)

        test_dataset = EEG2dDataset(test_X)
        test_dataloader = DataLoader(test_dataset, batch_size=math.ceil(len(test_dataset)/2), num_workers=1)

        with torch.no_grad():
            all_predictions = []
            for x in test_dataloader:
                preds = model(x.cuda())
                all_predictions.append(preds)
            predictions = torch.cat(all_predictions, dim=0)
            predictions_dict = {f"{sub}-{isplit}":predictions.tolist()}

        # 辞書をMLflowに保存
        mlflow_logger.experiment.log_dict(
            mlflow_logger.run_id, 
            dictionary=predictions_dict,
            artifact_file="test_predictions.json"
        )

