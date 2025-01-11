import pickle
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import mlflow

from eeg_utils import *

os.environ["no_proxy"]="localhost, 127.0.0.1, ::1"
mlflow_tracking_uri="http://localhost:5000"

class VideoDataset(Dataset):
    def __init__(self, video_array, labels, transform=None):
        '''
        video_array : (N,T,H,W)
        labels : (N,)
        output : (N,C,T,H,W)
        '''
        video_array = video_array[:, None, :, :, :]
        # 3チャンネルをコピー
        video_array = np.concatenate([video_array]*3, axis=1)
        self.video_array = torch.tensor(video_array, dtype=torch.float32)
        self.labels = torch.tensor(labels % 10 - 1)
        self.transform = transform

    def __len__(self):
        return len(self.video_array)

    def __getitem__(self, idx):
        T = self.video_array.size(2)
        if T > 250:
            start_idx = np.random.randint(T - 250 + 1)
            end_idx = start_idx + 250
            return self.video_array[idx, :, start_idx:end_idx], self.labels[idx]
        return self.video_array[idx], self.labels[idx]

class VideoDataset1(Dataset):
    def __init__(self, video_array, labels, transform=None):
        '''
        video_array : (N,T,H,W)
        labels : (N,)
        output : (N,C,T,H,W)
        '''
        video_array = video_array[:, None, :, :, :]
        # 1チャンネルのまま
        self.video_array = torch.tensor(video_array, dtype=torch.float32)
        self.labels = torch.tensor(labels % 10 - 1)
        self.transform = transform

    def __len__(self):
        return len(self.video_array)

    def __getitem__(self, idx):
        T = self.video_array.size(2)
        if T > 250:
            start_idx = np.random.randint(T - 250 + 1)
            end_idx = start_idx + 250
            return self.video_array[idx, :, start_idx:end_idx], self.labels[idx]
        return self.video_array[idx], self.labels[idx]

class VideoDataset2(Dataset):
    def __init__(self, video_array, labels, transform=None):
        '''
        video_array : (N,T,H,W)
        labels : (N,)
        output : (N,C,T,H,W)
        '''
        N,T,H,W = video_array.shape
        video_array = video_array[:, None, :, : ,:]
        # 3タイムステップを3チャンネルに
        video_array = np.concatenate([video_array[:, :, i:3*(T//3):3] for i in range(3)], axis=1)
        self.video_array = torch.tensor(video_array, dtype=torch.float32)
        self.labels = torch.tensor(labels % 10 - 1)
        self.transform = transform

    def __len__(self):
        return len(self.video_array)

    def __getitem__(self, idx):
        T = self.video_array.size(2)
        if T > 83:
            start_idx = np.random.randint(T - 83 + 1)
            end_idx = start_idx + 83
            return self.video_array[idx, :, start_idx:end_idx], self.labels[idx]
        return self.video_array[idx], self.labels[idx]

class VideoClassificationModel(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-5, weight_decay=1e-0, model_name="r3d_18"):
        super(VideoClassificationModel, self).__init__()
        self.save_hyperparameters()
        if model_name == "r3d_18":
            self.model = models.video.r3d_18(weights="DEFAULT")
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == "mc3_18":
            self.model = models.video.mc3_18(weights="DEFAULT")
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == "r2plus1d_18":
            self.model = models.video.r2plus1d_18(weights="DEFAULT")
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # elif model_name == "s3d":
        #     self.model = models.video.s3d(weights="DEFAULT")
        #     self.model.classifier[1] = nn.Conv3d(1024, num_classes, kernel_size=1, stride=1)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

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
      

import sys

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = "r3d_18"  # デフォルトのモデル名

max_epochs = dict(
    r3d_18=50,
    mc3_18=50,
    r2plus1d_18=30,
)[model_name]
learning_rate = dict(
    r3d_18=1e-5,
    mc3_18=1e-5,
    r2plus1d_18=1e-5,
)[model_name]
weight_decay = dict(
    r3d_18=3e-1,
    mc3_18=1e-1,
    r2plus1d_18=1e-2,
)[model_name]

# データの準備
_, target = prepare_all()

with open('topo32_2.pkl', 'rb') as f:
    train_data = pickle.load(f)

for sub in "0000 0001 0002 0003 0004".split():
    kf=KFold(3)
    for isplit, (train_index, val_index) in enumerate(kf.split(train_data[sub])):
        seed_everything(1, workers=True)

        train_X = np.concatenate([train_data[sub][i] for i in train_index], axis=0)
        val_X = train_data[sub][val_index[0]][:, 50:50 + 250]

        train_y = np.concatenate([target[sub][i] for i in train_index], axis=0)
        val_y = target[sub][val_index[0]]

        channel_max = train_X.max(axis=(0, 1, 2, 3), keepdims=True)
        train_X = train_X / channel_max
        val_X = val_X / channel_max

        train_dataset = VideoDataset(train_X, train_y)
        val_dataset = VideoDataset(val_X, val_y)

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)

        model = VideoClassificationModel(num_classes=3, learning_rate=learning_rate, weight_decay=weight_decay, model_name=model_name)

        # MLflow Loggerの設定
        cond=f"{model_name}_3ch"
        mlflow_logger = MLFlowLogger(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=model_name+"_2",
            run_name=f"{cond} {sub}-{isplit}",
            log_model = True,
        )
        mlflow_logger.log_hyperparams({
            "condition":cond,
            "sub":sub,
            "isplit":isplit,
            "batch_size":train_dataloader.batch_size,
        })

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=None,
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator='gpu',
            callbacks=[checkpoint_callback],
            deterministic=True,
            logger=mlflow_logger,
            log_every_n_steps=len(train_dataloader),
            default_root_dir=None,
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        local_path = mlflow.artifacts.download_artifacts(
            run_id=mlflow_logger.run_id,
            artifact_path="model/checkpoints/best-checkpoint/best-checkpoint.ckpt"
        )
        model = VideoClassificationModel.load_from_checkpoint(local_path, model_name=model_name)
        model.eval()
        model=model.cuda()

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
        with open('topo32_2_test.pkl', 'rb') as f:
            test_data = pickle.load(f)
        test_X = test_data[sub][0]
        test_X = test_X /channel_max
        dummy_y = np.ones(len(test_X), dtype=int)

        test_dataset=VideoDataset(test_X, dummy_y)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)

        model.eval()
        model=model.cuda()
        with torch.no_grad():
            all_predictions = []
            for x,label in test_dataloader:
                preds = model(x.cuda())
                all_predictions.append(preds)
            predictions = torch.cat(all_predictions, dim=0)
            predictions_dict = {f"{sub}-{isplit}":predictions.tolist()}
            pred_y = torch.argmax(predictions, 1)+1

        # 辞書をMLflowに保存
        mlflow_logger.experiment.log_dict(
            run_id=mlflow_logger.run_id,
            dictionary=predictions_dict,
            artifact_file="test_predictions.json"
        )
