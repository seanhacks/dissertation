import os 
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


# adapted from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/dataset.py
class MGDataset(Dataset):
    """
    This class allows for the fetching of the images and their corresponding masks.
    Also, it applies any transformations to both the images and the masks.
    In addition to that, it normalises mask values to 0 and 1, 0 being normal tissue/background
    and 1 being a suspicious mask/calcification.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # open images and mask to np arrays
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)

        # apply transformations.
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

            # normalise mask values.
            mask[mask < 128.0] = 0.0
            mask[mask >= 128.0] = 1.0

        return image, mask

if __name__ == "__main__":
    """
    This code simply tests that the images and corresponding masks are correct. 
    Also ensures that the standardisation of images, by size and normalisation, 
    work.
    """

    train_transform = A.Compose(
    [
        A.Resize(256,256),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ]
    )

    valid_transform = A.Compose(
    [
        A.Resize(256,256),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ]
    )

    train_dataset = MGDataset("../images/train_images/", "../images/train_masks/", train_transform)
    valid_dataset = MGDataset("../images/valid_images/", "../images/valid_masks/", valid_transform)

    def visualize_augmentations(dataset, idx=0, samples=1):
        dataset = copy.deepcopy(dataset)
        dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
        figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 10))
        image, mask = dataset[idx]

        # count how many ones in the set
        print(np.sum(mask))

        # graphs
        ax[0].imshow(image)
        ax[1].imshow(mask, interpolation="nearest")
        ax[0].set_title("Augmented image")
        ax[1].set_title("Augmented mask")
        plt.savefig("./augmented_image.png")

    random.seed(42)
    visualize_augmentations(valid_dataset, idx=12)


