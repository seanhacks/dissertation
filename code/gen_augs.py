import os 
import glob
import cv2 
import numpy as np
import pandas as pd 
import itertools

# The code generates the different augmentations that will be used for 
# the investigations.

IMAGES = "./images/train_images/"
MASKS = "./images/train_masks/"

# The following functions are used to generate the augmented images.

def flip_images_hor(input_folder, output_folder, include_original : bool = True):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    for image_file in image_files:
        original_image = cv2.imread(image_file)
        flipped_image = cv2.flip(original_image, 1)
        flipped_image_path = os.path.join(output_folder, f"hflip_{os.path.basename(image_file)}")
        if include_original:
            original_image_path = os.path.join(output_folder, os.path.basename(image_file))
            cv2.imwrite(original_image_path, original_image)
        cv2.imwrite(flipped_image_path, flipped_image)

def flip_images_ver(input_folder, output_folder, include_original : bool = True):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    for image_file in image_files:
        original_image = cv2.imread(image_file)
        flipped_image = cv2.flip(original_image, 0)
        flipped_image_path = os.path.join(output_folder, f"vflip_{os.path.basename(image_file)}")
        if include_original:
            original_image_path = os.path.join(output_folder, os.path.basename(image_file))
            cv2.imwrite(original_image_path, original_image)
        cv2.imwrite(flipped_image_path, flipped_image)

def rotate_image(input_folder, output_folder, angle, include_original : bool = True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    for image_file in image_files:
        for i in range(-2, 3, 1):
            if i != 0:
                original_image = cv2.imread(image_file)
                M = cv2.getRotationMatrix2D((original_image.shape[1]//2,original_image.shape[0]//2), i * angle, 1.0)
                #rotated_image = cv2.rotate(original_image, i * angle)
                rotated_image = cv2.warpAffine(original_image, M, (original_image.shape[1],original_image.shape[0]))
                file_path = os.path.join(output_folder, f"rot_{i}_{angle}_{os.path.basename(image_file)}")
                cv2.imwrite(file_path, rotated_image)
        if include_original:
            original_image_path = os.path.join(output_folder, os.path.basename(image_file))
            cv2.imwrite(original_image_path, original_image)

def translate_image(input_folder, output_folder, x_distance, include_original : bool = True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    for image_file in image_files:
        original_image = cv2.imread(image_file)
        M = np.float32([
            [1, 0, x_distance if "_L_" in image_file else -1 * x_distance],
            [0, 1, 0]
        ])
        shifted_image = cv2.warpAffine(original_image, M, (original_image.shape[1], original_image.shape[0]))
        file_path = os.path.join(output_folder, f"translate_x_{x_distance}_{os.path.basename(image_file)}")
        if include_original:
            original_image_path = os.path.join(output_folder, os.path.basename(image_file))
            cv2.imwrite(original_image_path, original_image)
        cv2.imwrite(file_path, shifted_image)

def shear_image_x(input_folder, output_folder, shear, include_original : bool = True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    for image_file in image_files: 
        original_image = cv2.imread(image_file)
        M = np.float32([
            [1, shear, 0],
            [0, 1, 0]
        ])
        M[0,2] = -M[0,1] * original_image.shape[1]//2
        M[1,2] = -M[1,0] * original_image.shape[0]//2
        sheared_image = cv2.warpAffine(original_image, M, (original_image.shape[1], original_image.shape[0]))
        file_path = os.path.join(output_folder, f"shear_x_{shear}_{os.path.basename(image_file)}")
        if include_original:
            original_image_path = os.path.join(output_folder, os.path.basename(image_file))
            cv2.imwrite(original_image_path, original_image)
        cv2.imwrite(file_path, sheared_image)

def shear_image_y(input_folder, output_folder, shear, include_original : bool = True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    for image_file in image_files: 
        original_image = cv2.imread(image_file).astype(np.float32)
        M = np.float32([
            [1, 0, 0],
            [shear, 1, 0]
        ])
        M[0,2] = -M[0,1] * original_image.shape[1]//2
        M[1,2] = -M[1,0] * original_image.shape[0]//2
        sheared_image = cv2.warpAffine(original_image, M, (original_image.shape[1], original_image.shape[0]))
        file_path = os.path.join(output_folder, f"shear_y_{shear}_{os.path.basename(image_file)}")
        if include_original:
            original_image_path = os.path.join(output_folder, os.path.basename(image_file))
            cv2.imwrite(original_image_path, original_image)
        cv2.imwrite(file_path, sheared_image)

def read_csv(f):
    file = f
    df = pd.read_csv(file)
    
    data = {}

    for index, row in df.iterrows():
        filename = str(row["Image_name"])
        classification = row["Pathology Classification/ Follow up"]
        data[str(filename) + ".jpg"] = str(classification)

    return data


def individual_aug_files():
    flip_images_hor(IMAGES, "./images/train_images_hflip/")
    flip_images_hor(MASKS, "./images/train_masks_hflip/")
    flip_images_ver(IMAGES, "./images/train_images_vflip/")
    flip_images_ver(MASKS, "./images/train_masks_vflip/")

    rotate_image(IMAGES, "./images/train_images_rotate_5/", 5)
    rotate_image(MASKS, "./images/train_masks_rotate_5/", 5)
    rotate_image(IMAGES, "./images/train_images_rotate_10/", 10)
    rotate_image(MASKS, "./images/train_masks_rotate_10/", 10)
    rotate_image(IMAGES, "./images/train_images_rotate_20/", 20)
    rotate_image(MASKS, "./images/train_masks_rotate_20/", 20)
    rotate_image(IMAGES, "./images/train_images_rotate_45/", 45)
    rotate_image(MASKS, "./images/train_masks_rotate_45/", 45)
    rotate_image(IMAGES, "./images/train_images_rotate_60/", 60)
    rotate_image(MASKS, "./images/train_masks_rotate_60/", 60)
    rotate_image(IMAGES, "./images/train_images_rotate_90/", 90)
    rotate_image(MASKS, "./images/train_masks_rotate_90/", 90)

    translate_image(IMAGES, "./images/train_images_translate_25/", 25)
    translate_image(MASKS, "./images/train_masks_translate_25/", 25)
    translate_image(IMAGES, "./images/train_images_translate_50/", 50)
    translate_image(MASKS, "./images/train_masks_translate_50/", 50)
    translate_image(IMAGES, "./images/train_images_translate_75/", 75)
    translate_image(MASKS, "./images/train_masks_translate_75/", 75)
    translate_image(IMAGES, "./images/train_images_translate_100/", 100)
    translate_image(MASKS, "./images/train_masks_translate_100/", 100)

    shear_image_x(IMAGES, "./images/train_images_shear_x_0.1/", 0.1)
    shear_image_x(MASKS, "./images/train_masks_shear_x_0.1/", 0.1)
    shear_image_x(IMAGES, "./images/train_images_shear_x_0.25/", 0.25)
    shear_image_x(MASKS, "./images/train_masks_shear_x_0.25/", 0.25)
    shear_image_x(IMAGES, "./images/train_images_shear_x_0.5/", 0.5)
    shear_image_x(MASKS, "./images/train_masks_shear_x_0.5/", 0.5)
    shear_image_x(IMAGES, "./images/train_images_shear_x_0.75/", 0.75)
    shear_image_x(MASKS, "./images/train_masks_shear_x_0.75/", 0.75)
    shear_image_x(IMAGES, "./images/train_images_shear_x_1/", 1)
    shear_image_x(MASKS, "./images/train_masks_shear_x_1/", 1)
    
    shear_image_y(IMAGES, "./images/train_images_shear_y_0.1/", 0.1)
    shear_image_y(MASKS, "./images/train_masks_shear_y_0.1/", 0.1)
    shear_image_y(IMAGES, "./images/train_images_shear_y_0.25/", 0.25)
    shear_image_y(MASKS, "./images/train_masks_shear_y_0.25/", 0.25)
    shear_image_y(IMAGES, "./images/train_images_shear_y_0.5/", 0.5)
    shear_image_y(MASKS, "./images/train_masks_shear_y_0.5/", 0.5)
    shear_image_y(IMAGES, "./images/train_images_shear_y_0.75/", 0.75)
    shear_image_y(MASKS, "./images/train_masks_shear_y_0.75/", 0.75)
    shear_image_y(IMAGES, "./images/train_images_shear_y_1/", 1)
    shear_image_y(MASKS, "./images/train_masks_shear_y_1/", 1)
    
    
def combined_aug_files():
    # Creates the combined augmentation datasets using the individual augmentation images.
    # Ensure that the individual augs files have been created first.
    tt = [i for i in itertools.product([0,1], repeat=5)]
    tt = [t for t in tt if sum(t) >= 2]
    augs = ["rot", "hflip", "shear_x", "shear_y", "trainslate"]
    values = [10, 0, 1.0, 0.75, 50]

    for row in tt:
       filename = make_filename(row, augs, values)
       make_file_with_augs(filename, row, augs, values)
       

def make_filename(row, augs, values):
    filename = ""
    for i in range(0,5):
        if row[i] == 1:
            filename += "_" + augs[i] + str(values[i])

    return filename

def make_file_with_augs(filename, row, augs, values):
    first=True
    for i in range(0,5):
        if row[i] == 1:
            if i == 0:
                rotate_image(IMAGES, "./images/train_images"+filename, 90, include_original=first)
                rotate_image(MASKS, "./images/train_masks" + filename, 90, include_original=first)
            elif i == 1:
                flip_images_ver(IMAGES, "./images/train_images" + filename, include_original=first)
                flip_images_ver(MASKS, "./images/train_masks" + filename, include_original=first)
            elif i == 2:
                shear_image_x(IMAGES, "./images/train_images" + filename, 0.5, include_original=first)
                shear_image_x(MASKS, "./images/train_masks" + filename, 0.5, include_original=first)
            elif i == 3: 
                shear_image_y(IMAGES, "./images/train_images" + filename, 0.25, include_original=first)
                shear_image_y(MASKS, "./images/train_masks" + filename, 0.25, include_original=first)
            elif i == 4:
                translate_image(IMAGES, "./images/train_images" + filename, 100, include_original=first)
                translate_image(MASKS, "./images/train_masks" + filename, 100, include_original=first)

            first=False

if __name__ == "__main__":
    individual_aug_files()
    combined_aug_files()
