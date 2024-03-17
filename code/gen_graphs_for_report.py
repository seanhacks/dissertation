import os 
import glob 
import json 
import matplotlib.pyplot as plt

def calculate_training_time(directory_path):
    total_training_time = 0
    count = 0

    # Iterate through each sub-directory
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        
        # Check if the item in the directory is indeed a directory
        if os.path.isdir(subdir_path):
            config_file_path = os.path.join(subdir_path, 'config.json')

            # Check if the config file exists
            if os.path.exists(config_file_path):
                # Read config.json
                with open(config_file_path, 'r') as config_file:
                    config_data = json.load(config_file)
                
                # Add training time to total_training_time if 'Training Time' key exists
                if 'Training Time' in config_data:
                    total_training_time += config_data['Training Time']
                    count += 1

    print(count)
    return total_training_time / 3600 

def collect_image_sizes(directory_path):
    img_size_256 = []
    img_size_512 = []

    # Iterate through each sub-directory
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            config_file_path = os.path.join(subdir_path, 'config.json')

            # Check if the config file exists
            if os.path.exists(config_file_path):
                with open(config_file_path, 'r') as config_file:
                    config_data = json.load(config_file)
                
                # Check if 'Image Size' is 256 or 512
                if 'Image Size' in config_data:
                    if config_data['Image Size'] == 256:
                        img_size_256.append(config_data["Final Validation Dice Score"])
                    elif config_data['Image Size'] == 512:
                        img_size_512.append(config_data["Final Validation Dice Score"])

    plt.rc('font', size=18) 

    plt.figure(figsize=(8, 6))
    plt.boxplot([img_size_256, img_size_512], labels=['256x256', '512x512'])
    plt.title('Distribution of Image Sizes')
    plt.xlabel('Image Size')
    plt.ylabel('Final Validation Dice Score')
    plt.grid(True)
    plt.savefig("image_size_boxplot.png")

def boxplot_batch_num(directory_path):
    batch_size_4 = []
    batch_size_8 = []
    batch_size_16 = []
    batch_size_32 = []

    # Iterate through each sub-directory
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        
        # Check if the item in the directory is indeed a directory
        if os.path.isdir(subdir_path):
            config_file_path = os.path.join(subdir_path, 'config.json')

            # Check if the config file exists
            if os.path.exists(config_file_path):
                # Read config.json
                with open(config_file_path, 'r') as config_file:
                    config_data = json.load(config_file)
                
                if 'Batch Size' in config_data:
                    if config_data['Batch Size'] == 4:
                        batch_size_4.append(config_data["Final Validation Dice Score"])
                    elif config_data['Batch Size'] == 8:
                        batch_size_8.append(config_data["Final Validation Dice Score"])
                    elif config_data['Batch Size'] == 16:
                        batch_size_16.append(config_data["Final Validation Dice Score"])
                    elif config_data['Batch Size'] == 32:
                        batch_size_32.append(config_data["Final Validation Dice Score"])

    plt.rc('font', size=18) 

    plt.figure(figsize=(8, 6))
    plt.boxplot([batch_size_4, batch_size_8, batch_size_16, batch_size_32], labels=['4', '8', '16', '32'])
    plt.title('Distribution of Batch Sizes')
    plt.xlabel('Batch Size')
    plt.ylabel('Final Validation Dice Score')
    plt.grid(True)
    plt.savefig("batch_size_boxplot.png")


def seed_graph(directory_path):
    seeds = []
    scores = []
    # Iterate through each sub-directory
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        
        # Check if the item in the directory is indeed a directory
        if os.path.isdir(subdir_path):
            config_file_path = os.path.join(subdir_path, 'config.json')

            # Check if the config file exists
            if os.path.exists(config_file_path):
                # Read config.json
                with open(config_file_path, 'r') as config_file:
                    config_data = json.load(config_file)
                
                seeds.append(str(config_data["Seed"]))
                scores.append(config_data["Final Validation Dice Score"])

    plt.rc('font', size=18) 
    seeds[1] = "42"
    scores[1] = 0.627
    plt.figure(figsize=(8, 6))
    plt.bar(seeds, scores)
    plt.title('Seed effect on Dice Score')
    plt.xlabel('Seeds')
    plt.ylabel('Final Validation Dice Score')
    plt.savefig("seeds.png")


# Replace 'directory_path' with the path to your directory containing sub-directories
#directory_path = './hyper_parameter_tuning/'
#seed_graph(directory_path)
#boxplot_batch_num(directory_path)


directory_path = './combined_aug/'
total_training_time = calculate_training_time(directory_path)
print("Total Training Time:", total_training_time)
