import os 
from matplotlib import pyplot as plt
import re
import pandas as pd
import cv2

def read_csv(f):
    """
    Reads the csv containing information on all the images.
    gets the class (normal/benign/malignant) of each image.
    """
    file = f
    df = pd.read_csv(file)
    
    data = {}

    for index, row in df.iterrows():
        filename = str(row["Image_name"])
        classification = row["Pathology Classification/ Follow up"]
        data[str(filename) + ".jpg"] = str(classification)

    return data

def count_dist_per_subset(data):
    """
    Counts the number of images and how many of each class are in
    each of the subsets (training, validation, testing).
    """

    DIRS = ["./images/train_images", "./images/valid_images", "./images/test_images"]
    ORDER = ["train", "valid", "test"]
    res = {
        "train" : [0,0,0],
        "valid" : [0,0,0],
        "test" : [0,0,0],
        "total" : 0
    }

    files_list = []
    for d in range(len(DIRS)):
        for root, dirs, files in os.walk(DIRS[d]):
            print(len(files))
            for filename in files:
                #key = filename[0:filename.index("_")]+".txt"
                key = filename
                files_list.append(key)
                try:
                    if data[key] == "Normal":
                        res[ORDER[d]][0] += 1
                        res["total"] += 1
                    elif data[key] == "Benign":
                        res[ORDER[d]][1] += 1
                        res["total"] += 1
                    elif data[key] == "Malignant":
                        res[ORDER[d]][2] += 1
                        res["total"] += 1
                    else: 
                        print("something wrong")
                except Exception:
                    print(filename)
                    continue

    print(res)
    print("Train: " + str(res["train"][0]/sum(res["train"])) + ", " + str(res["train"][1]/sum(res["train"])) + ", " + str(res["train"][2]/sum(res["train"])))
    print("Validation: " + str(res["valid"][0]/sum(res["valid"])) + ", " + str(res["valid"][1]/sum(res["valid"])) + ", " + str(res["valid"][2]/sum(res["valid"])))
    print("Test: " + str(res["test"][0]/sum(res["test"])) + ", " + str(res["test"][1]/sum(res["test"])) + ", " + str(res["test"][2]/sum(res["test"])))



count_dist_per_subset(read_csv("./dataset_info/radiology_annotations.csv"))