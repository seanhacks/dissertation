import torch 
import sys
from dataset import MGDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from model import UNet
from utils import MetricTracker
import numpy as np
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import set_seed

# This allows a trained model to be evaluated on the test set.
# outputting the average dice score on the test set images.
# as well as the other metrics such as precision, and recall.

DEVICE = 'cuda'

def evaluate(loader, model):
    """
    Get the evaluation metrics for the model's performance on the
    test dataset. These include:
    - Dice score
    - Accuracy
    - Recall
    - Precision
    - Time Taken (s)
    """

    num_correct = 0
    num_pixels  = 0
    dice_score  = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            y = y.cpu().numpy()
            preds = preds.cpu().numpy()

            tp += np.sum(np.logical_and(y == 1, preds == 1))
            tn += np.sum(np.logical_and(y == 0, preds == 0))
            fp += np.sum(np.logical_and(y == 0, preds == 1))
            fn += np.sum(np.logical_and(y == 1, preds == 0))

            dice_score += (2 * (preds * y).sum()) / (2 * (preds * y).sum()+ ((preds*y)<1).sum())
            dice_score += ((2.*(preds*y).sum())+1e-6)/(preds.sum()+y.sum()+1e-6)

    end_time = time.time()
    metrics = MetricTracker()
    metrics.addMetrics( dice_score/len(loader), tp, tn , fn, fp )

    return end_time-start_time, metrics

if __name__ == "__main__":

    set_seed(42)

    test_transform_first = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.Normalize(
                mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5],
            ),
            ToTensorV2(),
        ]
    )

    test_ds_first_set = MGDataset(
        image_dir = "./images/test_images",
        mask_dir = "./images/test_masks",
        transform=test_transform_first
    )

    first_loader = DataLoader(
        test_ds_first_set,
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        shuffle=False
    )

    model_directory = sys.argv[1]
    model_first = UNet(n_channels=3, n_classes=1).to(DEVICE)
    checkpoint1 = torch.load(model_directory, map_location=DEVICE)
    model_first.load_state_dict(checkpoint1['state_dict'])

    results = evaluate(first_loader, model_first)
    print(results[1])
    print(results[0])
