import json
import os
import numpy as np
import PIL.Image
import cv2
import csv

import shutil
 
def csv_to_json(csv_file_path, json_file_path):
    """
    Converts the segmentation file to a json file. That we will use when constructing
    the ground truth masks.
    """
    data_dict = {}
 
    with open(csv_file_path, encoding = 'utf-8') as csv_file_handler:
        csv_reader = csv.DictReader(csv_file_handler)
        prev_key = None 
        row_arr = []

        for rows in csv_reader:

            key = rows['#filename']
            if "_DM_" in key:
                continue 

            if prev_key != None and key != prev_key: 
                data_dict[prev_key] = row_arr
                row_arr = []


            row = json.loads(rows["region_shape_attributes"])
            shape = row["name"]

            if shape == "polygon" or shape == "polyline":
                row_arr.append({"name":row["name"], "all_points_x": row["all_points_x"], "all_points_y": row["all_points_y"]})
            elif shape == "circle":
                row_arr.append({"name":row["name"], "cx": row["cx"], "cy": row["cy"], "r": row["r"]})
            elif shape == "ellipse":
                row_arr.append({"name":row["name"], "cx": row["cx"], "cy": row["cy"], "rx": row["rx"], "ry": row["ry"]})

            prev_key = key
 
    with open(json_file_path, 'w', encoding = 'utf-8') as json_file_handler:
        json_file_handler.write(json.dumps(data_dict, indent = 4))
 
csv_to_json("./dataset_info/Radiology_hand_drawn_segmentations_v2.csv", "./dataset_info/segmentations.json")

# Generates the segmentation masks from the coordinates given in the csv.

JSON_FILE = "./dataset_info/segmentations.json" 
IMAGES_DIRECTORY = "./images/all/"

with open(JSON_FILE, "r") as read_file:
    data = json.load(read_file)

all_file_names = os.listdir(IMAGES_DIRECTORY)

shutil.rmtree("./images/masks")
os.makedirs("./images/masks/")

for j in range(len(all_file_names)): 
    image_name=all_file_names[j]

    img = np.asarray(PIL.Image.open(IMAGES_DIRECTORY + image_name))
    
    if image_name in data:
        masks = []

        # loop over all of the segmentation shapes of the image
        for i in range(len(data[all_file_names[j]])):
            mask = np.zeros((img.shape[0],img.shape[1]))
            if (data[all_file_names[j]][i]['name'] == 'polygon'
                or data[all_file_names[j]][i]['name'] == 'polyline'): 
                shape1_x=data[all_file_names[j]][i]['all_points_x']
                shape1_y=data[all_file_names[j]][i]['all_points_y']
                ab=np.stack((shape1_x, shape1_y), axis=1)

                mask=cv2.drawContours(mask, [ab], -1, 255, -1)
                masks.append(mask)

            elif data[all_file_names[j]][i]['name'] == 'circle':
                ab=np.stack((shape1_x, shape1_y), axis=1)
                mask = np.zeros((img.shape[0],img.shape[1]))
                cx = data[all_file_names[j]][i]['cx']
                cy = data[all_file_names[j]][i]['cy']
                r = data[all_file_names[j]][i]['r']

                mask=cv2.circle(mask, (cx, cy), r, 255, -1)
                masks.append(mask)

            elif data[all_file_names[j]][i]['name'] == 'ellipse':
                ab=np.stack((shape1_x, shape1_y), axis=1)
                mask = np.zeros((img.shape[0],img.shape[1]))
                cx = data[all_file_names[j]][i]['cx']
                cy = data[all_file_names[j]][i]['cy']
                rx = data[all_file_names[j]][i]['rx']
                ry = data[all_file_names[j]][i]['ry']

                mask=cv2.ellipse(mask, (cx, cy), (rx,ry),0,0,360, 255, -1)
                masks.append(mask)

            else: 
                pass

        h, w = masks[0].shape
        result = np.zeros((h,w))
        for mask in masks:
            result = cv2.add(result, mask)
        cv2.imwrite('./images/masks/' +  image_name, result.astype(np.uint8))

    else:
        mask = np.zeros((img.shape[0],img.shape[1]))
        cv2.imwrite('./images/masks/' +  image_name, mask.astype(np.uint8))





