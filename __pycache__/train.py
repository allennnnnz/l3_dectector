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
from pytorch_lightning.accelerators import MPSAccelerator

CHECKPOINT_PATH = "./saved_models"
class_num = [30687,4999,6161,4142,5703,3458,5372,2597,5420,8368,5467,6271]
NUM_WORKERS = os.cpu_count()


def train_simclr(batch_size, max_epochs=500, **kwargs):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
        accelerator="mps" if torch.backends.mps.is_available() else 'cpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ]
    )
    trainer.logger._default_hp_metric = False  # 可選的日誌參數

    pretrained_filename ="/Users/allen/扣得/l3_dectector/saved_models/SimCLR/lightning_logs/version_35/checkpoints/epoch=1-step=3544.ckpt"
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

def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=400, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
                         accelerator="cpu" ,
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True,
                         check_val_every_n_epoch=5)
    trainer.logger._default_hp_metric = False
    
    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True, 
                                   drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = data.DataLoader(test_feats_data, batch_size=batch_size, shuffle=False, 
                                  drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

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
    
    simclr_model = train_simclr(
        batch_size=50,
        hidden_dim=256,
        lr=5e-3,
        temperature=0.07,
        weight_decay=1e-4,
        max_epochs=4
    )
    torch.save(simclr_model.state_dict(),'/Users/allen/扣得/l3_dectector/models/sim_CLR.pth')
    
    train_feats_simclr = prepare_data_features(simclr_model, train_all_dataset)
    test_feats_simclr = prepare_data_features(simclr_model, test_all_dataset)
    
    results = {}
    compelete_model, all_results = train_logreg(batch_size=60,
                                        train_feats_data=train_feats_simclr,
                                        test_feats_data=test_feats_simclr,
                                        model_suffix='all',
                                        feature_dim=train_feats_simclr.tensors[0].shape[1],
                                        num_classes=12,
                                        lr=1e-3,
                                        weight_decay=1e-3,
                                        class_num=class_num) 
    
    torch.save(compelete_model.state_dict(),'/Users/allen/扣得/l3_dectector/models/linear.pth')
    
    test_scores = all_results["test"] 
    print(f'Test accuracy : {100*test_scores:4.2f}%')