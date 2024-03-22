import pandas as pd
import os 
import random

csv = pd.read_csv("./dataset_info/radiology_annotations.csv")

# Removes the unwanted images from the dataset, and then splits the images
# into a 70-15-15 training, validation, testing split.

# all of the image tags that we will keep in the dataset.
good = ["suspicious nodules", "malignant", "diffuse macrocalcifications", "malignant mass", "microcalcifications", 
        "benign macrocalcification", "suspisious mass", "malibnant mass", "macrocalcifications", "benign macrocalcifications",
        "microcalcification", "malignant microcalcifications", "masses", "normal", "calcifications", "suspicious masses", 
        "calcification", "macrocalcification", "suspicious mass", "benign mas", "benign mass", "benign masses", "suspicous mass"]

IMG_DIR = "./images/all/"
MASK_DIR = "./images/masks/"
TRAIN_IMG_DIR = "./images/train_images/"
TRAIN_MASK_DIR = "./images/train_masks/"
VAL_IMG_DIR = "./images/valid_images/"
VAL_MASK_DIR = "./images/valid_masks/"
TEST_IMG_DIR =  "./images/test_images/"
TEST_MASK_DIR = "./images/test_masks/"

to_delete = []


# delete unwanted images.
for index, row in csv.iterrows():
    has_tag = False
    if "DM" in row["Image_name"]:
        csv = csv.drop([index])
        continue

    for tag in good:
        for tag2 in row["Tags"].split(","):
            if tag == tag2:
                has_tag = True 

    if not has_tag:
        to_delete.append(row["Image_name"])
        csv = csv.drop([index])

print(len(csv.index))

for f in to_delete:
    os.system(f"rm {IMG_DIR}{f}.jpg")
    os.system(f"rm {MASK_DIR}{f}.jpg")
    print("Image and Mask Deleted")

def count_classes(df):
    classes = [0, 0]
    for index, row in csv.iterrows():
        if row["Pathology Classification/ Follow up"] == "Normal":
            classes[0] += 1
        else:
            classes[1] += 1
        
    return classes 

arr = count_classes(csv)
print(arr)


def within_tolerance(lst, len_norm, len_sus, tol = 8):
    if len(lst[0]) in range(len_norm-tol, len_norm+tol+1) and len(lst[1]) in range(len_sus-tol, len_sus+tol+1):
        return True
    return False

def distribute(df, num_normal, num_sus):
    TOL =  3
    DATASET_SIZE = num_normal + num_sus
    TRAIN_NORM_SIZE = (int) (0.7 * num_normal)
    TRAIN_SUS_SIZE = (int) (0.7 * num_sus)
    VAL_NORM_SIZE = (int) (0.15 * num_normal)
    VAL_SUS_SIZE = (int) (0.15 * num_sus)
    TEST_NORM_SIZE = (int) (0.15 * num_normal)
    TEST_SUS_SIZE = (int) (0.15 * num_sus) 
    train = [[],[]]
    val = [[],[]]
    test = [[],[]]


    while ( not (within_tolerance(train, TRAIN_NORM_SIZE, TRAIN_SUS_SIZE, TOL) and within_tolerance(val, VAL_NORM_SIZE, VAL_SUS_SIZE, TOL) and within_tolerance(test, TEST_NORM_SIZE, TEST_SUS_SIZE, TOL)) ):
        train = [[],[]]
        val = [[],[]]
        test = [[],[]]
        for i in range(DATASET_SIZE):
            num = random.random()

            if num < 0.7:
                if df.iloc[i]["Pathology Classification/ Follow up"] == "Normal":
                    train[0].append(df.iloc[i]["Image_name"])
                else: 
                    train[1].append(df.iloc[i]["Image_name"])
            elif num < 0.85:
                if df.iloc[i]["Pathology Classification/ Follow up"] == "Normal":
                    val[0].append(df.iloc[i]["Image_name"])
                else: 
                    val[1].append(df.iloc[i]["Image_name"])
            else:
                if df.iloc[i]["Pathology Classification/ Follow up"] == "Normal":
                    test[0].append(df.iloc[i]["Image_name"])
                else: 
                    test[1].append(df.iloc[i]["Image_name"])

        print("Failed to Split")

    print("Done with Split")
    return train, val, test


train, val, test = distribute(csv, arr[0], arr[1])

def move_to_files(imgs, set_):
    print("Moving Files")
    masks, all_ = "", ""
    if set_ == "train":
        masks = TRAIN_MASK_DIR
        all_ = TRAIN_IMG_DIR
        if not os.path.exists(TRAIN_IMG_DIR):
            os.mkdir(TRAIN_IMG_DIR)
        if not os.path.exists(TRAIN_MASK_DIR):
            os.mkdir(TRAIN_MASK_DIR)
    elif set_ == "val":
        masks = VAL_MASK_DIR
        all_ = VAL_IMG_DIR
        if not os.path.exists(VAL_IMG_DIR):
            os.mkdir(VAL_IMG_DIR)
        if not os.path.exists(VAL_MASK_DIR):
            os.mkdir(VAL_MASK_DIR)
    elif set_ == "test":
        masks = TEST_MASK_DIR
        all_ = TEST_IMG_DIR
        if not os.path.exists(TEST_IMG_DIR):
            os.mkdir(TEST_IMG_DIR)
        if not os.path.exists(TEST_MASK_DIR):
            os.mkdir(TEST_MASK_DIR)

    for im in imgs[0]:
        os.system(f"cp {IMG_DIR}{im}.jpg {all_}")
        os.system(f"cp {MASK_DIR}{im}.jpg {masks}")

    for im in imgs[1]:
        os.system(f"cp {IMG_DIR}{im}.jpg {all_}")
        os.system(f"cp {MASK_DIR}{im}.jpg {masks}")

    print("Done moving files")

move_to_files(train, "train")
move_to_files(val, "val")
move_to_files(test, "test")




