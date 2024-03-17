import torch 
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
#from model import UNET
from model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import (
    save_checkpoint,
    get_loaders, 
    check_accuracy,
    save_config_to_json, 
    save_results_to_csv,
    adjust_learning_rate_in_training,
    set_seed,
    DiceBCELoss,
    DiceLoss,
    TverskyLoss,
    EarlyStopper,
    MetricTracker
)
import utils as utils
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import time
import json
import csv
import segmentation_models_pytorch as smp

from pathlib import Path

# Training Pipeline adapted from: https://www.youtube.com/watch?v=IHq1t7NxS8k

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
VAL_IMG_DIR = "./images/valid_images"
VAL_MASK_DIR = "./images/valid_masks"

def train_fn(loader, model, optimizer, loss_fn, scalar):
    # Initialise loading bar for current epoch.
    loop = tqdm(loader)

    for b, (data, targets) in enumerate(loop):
        # Move input data and masks to the GPU 
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward step: simply performing the weights times input calculation.
        with torch.cuda.amp.autocast():
            data = data.type(torch.cuda.FloatTensor)
            targets = targets.type(torch.cuda.FloatTensor)
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward step: 
        optimizer.zero_grad()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(im_height, im_width, opt, lo_fn, starting_l, min_l, batch, 
         iterations, directory, train_images_dir, train_masks_dir, notes, seed,
         alpha=0, beta=0):
    
    # Set the seed for all RNG
    set_seed(seed)

    # Create directory if not already made
    path = Path(directory)
    path.mkdir(parents=True)

    # Start time of training in seconds after epoch
    time_start = time.time()

    # Create the data augmentations that will be performed on the training set.
    train_transform = A.Compose(
        [
            A.Resize(height=im_height, width=im_width),
            A.Normalize(
                mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5],
            ),
            ToTensorV2(),
        ]
    )
    
    # Create the augmentations that will be performed on validation set.
    val_transform = A.Compose(
        [
            A.Resize(height=im_height, width=im_width),
            A.Normalize(
                mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5],
            ),
            ToTensorV2(),
        ],
    )

    # Initialise model, loss function, and optimizer.
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    
    loss_fn = ""
    optimizer = ""
    if lo_fn == "Dice Loss":
        loss_fn = DiceLoss()
    elif lo_fn == "DiceBCE": 
        loss_fn = DiceBCELoss()
    elif lo_fn == "Tversky Loss":
        loss_fn = TverskyLoss(alpha=alpha, beta=beta)

    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=starting_l)
    elif opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=starting_l, momentum=0.9)
    elif opt == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=starting_l)
    elif opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=starting_l)

    early_stopper = EarlyStopper(patience=5,min_delta=0.03)
    train_metric = MetricTracker()
    valid_metric = MetricTracker()

    # Get Dataloader object for training and validation data.
    train_loader,val_loader = get_loaders(
        train_images_dir, 
        train_masks_dir, 
        VAL_IMG_DIR, 
        VAL_MASK_DIR, 
        batch, 
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    # Scaled gradients prevent underflowing gradients
    scalar = torch.cuda.amp.GradScaler()
    max_dice = 0
    stopped_epoch = 0

    for e in range(iterations):
        print(f'Epoch Number {e+1}: Learning Rate {optimizer.param_groups[0]["lr"]}')
        train_fn(train_loader, model, optimizer, loss_fn, scalar)

        check_accuracy(val_loader, model, False, valid_metric, device=DEVICE, loss_fn=loss_fn)
        check_accuracy(train_loader, model, True, train_metric, device=DEVICE, loss_fn=loss_fn)
        adjust_learning_rate_in_training(optimizer, e, iterations, starting_l, min_l)
        
        # if the current model has a greater dice score than the previous max, save the model.
        if valid_metric.dice[-1] > max_dice:
            max_dice = valid_metric.dice[-1]

            checkpoint = {
                "state_dict":model.state_dict(), 
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"./{directory}/max_dice.pth.tar")

        if e == iterations-1 or early_stopper.early_stop(valid_metric.dice[-1], train_metric.dice[-1], e):
            checkpoint = {
                "state_dict":model.state_dict(), 
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"./{directory}/last_epoch.pth.tar")
            stopped_epoch = e
            break

        

    # TRAINING DONE, NOW WE HANDLE THE DATA.
            
    # Retrieve end time of training after epoch.
    time_end = time.time()

    # Calculate training time in seconds.
    training_time_secs = time_end - time_start
    
    # Put all data to be graphed in appropriate arrays to be put in Dataframes.
    all_accs = train_metric.accuracy + valid_metric.accuracy 
    all_precisions = train_metric.precision + valid_metric.precision
    all_recall = train_metric.recall + valid_metric.recall
    all_losses = train_metric.loss + valid_metric.loss
    epochs = [x for x in range(stopped_epoch + 1)] + [x for x in range(stopped_epoch + 1)]
    acc_type = ["Training"]*(stopped_epoch+1) + ["Validation"]*(stopped_epoch+1) 

    all_dices = train_metric.dice + valid_metric.dice  

    print(f"Length of dice metrics array: {len(all_dices)}")
    print(f"Length of accuracy metric array: {len(all_accs)}")
    print(f"Length of number of epochs: {len(epochs)}")
    print(f"Length of subtypes array: {len(acc_type)}")

    df = pd.DataFrame(epochs, columns=["Epochs"])
    df['Subset'] = acc_type
    df['Dice Score'] = all_dices
    df['Accuracy'] = all_accs
    df['Precision'] = all_precisions
    df['Recall'] = all_recall
    df['Loss'] = all_losses
    print(df)

    # Graph the dataframes
    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df, x="Epochs", y="Loss", hue="Subset")
    plt.savefig(f'./{directory}/loss.png')

    plt.figure()
    sns.lineplot(data=df, x="Epochs", y="Dice Score", hue="Subset")
    plt.savefig(f'./{directory}/dice.png')


    # Save configuration of model + training pipeline to json    
    save_config_to_json(im_height, batch, starting_l, min_l, stopped_epoch, train_metric, valid_metric, training_time_secs, directory, notes, seed)

    # Save all dice metrics and acc metrics for training and validation to csv file
    save_results_to_csv(directory, df)

    return valid_metric.dice[-1]
    
if __name__ == "__main__":
    STARTING_LR = 1e-4
    MIN_LR = 1e-5
    BATCH_SIZE = 4
    NUM_EPOCHS = 150
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    TRAIN_IMG_DIR = "./images/train_images"
    TRAIN_MASK_DIR = "./images/train_masks"

    main(IMAGE_HEIGHT, IMAGE_WIDTH, "adam", "Dice Loss", STARTING_LR, MIN_LR, BATCH_SIZE, NUM_EPOCHS, 
         "./test", TRAIN_IMG_DIR, TRAIN_MASK_DIR, "notes:...", 42)
    
