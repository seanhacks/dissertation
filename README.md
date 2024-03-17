## Steps To Run

1. Install the python libraries using the requirements.txt file.
2. Download the images and csv files from: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611
3. Arrange your file directory as follows:
```
├── dataset_info
│   ├── radiology_annotations.csv
│   ├── Radiology_hand_drawn_segmentations_v2.csv
├── images
│   ├── all
│   │   ├── *.jpg (all CESM images from dataset)
|   ├── masks
│   │   ├── (Empty for now)
├── *.py (all the python scripts)
├── requirements.txt
```
4. run the create_binary_masks.py file to generate the segmentation mask ground truths of each image.
5. run the partition_dataset.py file to split the data into a 70-15-15 split of training, validation, and testing data.
6. run train.py to being training your own model!

### Notes
- In order to change model parameters edit the if __name__ =="__main__" section in the train.py file.
- In order to generate the data augmentations used in the dissertation call the gen_augs.py file.
- In order to get image predictions with a trained model, use the predict.py file.
- In order to evaluate a trained model on the test set, use the evaluate.py file.
- Call the dataset_distribution.py file to see the dataset split after calling partition_dataset.py. This includes information such as how many images are in each subset and how many of each class (normal, benign, malignant) is in each subset.
