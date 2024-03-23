from copy import deepcopy
import os
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from argumentation import *
from data import *
import matplotlib.pyplot as plt
from simclr_model import *
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from classify_model import *
from feature_map import *

CHECKPOINT_PATH = "./saved_models"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 簡化 GPU 判斷
NUM_WORKERS = os.cpu_count()


def train_simclr(batch_size, max_epochs=500, **kwargs):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",  # 使用 GPU 加速
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ]
    )
    trainer.logger._default_hp_metric = True  # 可選的日誌參數

    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = SimCLR.load_from_checkpoint(pretrained_filename)
    else:
        try:
            train_loader = data.DataLoader(
                unlabeled_all_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                num_workers=NUM_WORKERS,
                persistent_workers=True
            )
            val_loader = data.DataLoader(
                contrast_all_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=NUM_WORKERS,
                persistent_workers=True
            )
            pl.seed_everything(42)
            model = SimCLR(max_epochs=max_epochs, **kwargs)
            trainer.fit(model, train_loader, val_loader)
            model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        except Exception as e:
            print(f"Exception occurred during training: {e}")

    return model

def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=100, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True,
                         check_val_every_n_epoch=10)
    trainer.logger._default_hp_metric = None
    
    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True, 
                                   drop_last=False, pin_memory=True, num_workers=4)
    test_loader = data.DataLoader(test_feats_data, batch_size=batch_size, shuffle=False, 
                                  drop_last=False, pin_memory=True, num_workers=4)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}
        
    return model, result

if __name__ == "__main__":
    
   

    train_feats_simclr = prepare_data_features(simclr_model, contrast_all_dataset)
    test_feats_simclr = prepare_data_features(simclr_model, test_all_dataset)
    
    results = {}
    _, all_results = train_logreg(batch_size=64,
                                        train_feats_data=train_feats_simclr,
                                        test_feats_data=test_feats_simclr,
                                        model_suffix='all',
                                        feature_dim=train_feats_simclr.tensors[0].shape[1],
                                        num_classes=12,
                                        lr=1e-3,
                                        weight_decay=1e-3)
    results = all_results
    
    dataset_sizes = sorted([k for k in results])
    test_scores = [results[k]["test"] for k in dataset_sizes]

    fig = plt.figure(figsize=(6,4))
    plt.plot(dataset_sizes, test_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
    plt.xscale("log")
    plt.xticks(dataset_sizes, labels=dataset_sizes)
    plt.title("CV classification over dataset size", fontsize=14)
    plt.xlabel("Number of images per class")
    plt.ylabel("Test accuracy")
    plt.minorticks_off()
    plt.show()

    print(f'Test accuracy : {100*results:4.2f}%')