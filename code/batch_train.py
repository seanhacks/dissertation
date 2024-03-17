from train import main
import random

# This file allows the running of all the different investigations run for the dissertation.


def hyperparameter_tuning():
    """
    The initial investigation of the hyperparameter tuning of the model.
    """

    STARTING_LR = [1e-3, 1e-4, 1e-5]
    MIN_LR      = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    BATCH_SIZE  = [4, 8, 16, 32]
    MAX_NUM_EPOCHS = 75 
    IMG_SIZE = [256, 512]
    TRAIN_IMG_DIR = "./images/train_images"
    TRAIN_MASK_DIR = "./images/train_masks"
    SEED = 42

    dice_scores = []
    count = 0
    for s_lr in STARTING_LR:
        for l_lr in MIN_LR:
            if s_lr >= l_lr:
                for b_size in BATCH_SIZE:
                    for im_size in IMG_SIZE:
                        if ( b_size == 32 or b_size == 16 ) and im_size == 512:
                            continue 
                        d = main(im_size, im_size, "adam", "Dice Loss", s_lr, l_lr, b_size, MAX_NUM_EPOCHS, 
                                 f"./hyper_parameter_tuning/{s_lr}_{l_lr}_{b_size}_{im_size}",
                                 TRAIN_IMG_DIR, TRAIN_MASK_DIR, "Tuning Hyperparameters", SEED)
                        count += 1
                        dice_scores.append((d, s_lr, l_lr, b_size, im_size))

    print(max(dice_scores, key=lambda x : x[0]))


def best_loss_and_optimizer():
    """
    Finds the best loss and optimiser function combination.
    """

    STARTING_LR = 1e-4
    MIN_LR      = 1e-5
    BATCH_SIZE  = 4
    MAX_NUM_EPOCHS = 75
    IMG_SIZE = 256
    TRAIN_IMG_DIR = "./images/train_images"
    TRAIN_MASK_DIR = "./images/train_masks"
    LOSS = ["Dice Loss", "DiceBCE", "Tversky Loss"]
    OPTIM = ["adam", "rmsprop", "adamax", ]
    dice_scores = []

    for l in LOSS:
        for o in OPTIM:
            d = main(IMG_SIZE, IMG_SIZE, o, l, STARTING_LR, MIN_LR, BATCH_SIZE, MAX_NUM_EPOCHS, 
                 f"./loss_fn_optim/{l}_{o}",
                 TRAIN_IMG_DIR, TRAIN_MASK_DIR, "Finding best loss and optim combo wombo.", 42, alpha=0.2, beta=0.8)
            
            dice_scores.append((d, l, o))

    print(max(dice_scores, key=lambda x : x[0]))          

def find_good_seed():
    """
    Sets 10 random seeds and sees which one is the best.
    """

    STARTING_LR = 1e-4
    MIN_LR      = 1e-5
    BATCH_SIZE  = 4
    MAX_NUM_EPOCHS = 75
    IMG_SIZE = 256
    TRAIN_IMG_DIR = "./images/train_images"
    TRAIN_MASK_DIR = "./images/train_masks"
    LOSS = "Dice Loss"
    OPTIM = "adam"

    seed_set = set()
    dice_scores = []

    for i in range(10):
        seed = random.randint(1,1000)
        print(seed)
        while seed in seed_set:
            seed = random.randint(1,1000)

        seed_set.add(seed)
        d = main(IMG_SIZE, IMG_SIZE, OPTIM, LOSS, STARTING_LR, MIN_LR, BATCH_SIZE, 
             MAX_NUM_EPOCHS, f"./finding_good_seed/seed_num_{seed}", TRAIN_IMG_DIR, 
             TRAIN_MASK_DIR, str("We are trying to find a good seed, this is seed " + str(seed)), seed)
    
        dice_scores.append((d, seed))

    print(max(dice_scores, key=lambda x : x[0]))
   
def individual_augs():
    """
    Finds the best parameters for each of the individual augmentations we are using.
    """

    STARTING_LR = 1e-4
    MIN_LR      = 1e-5
    BATCH_SIZE  = 4
    MAX_NUM_EPOCHS = 75
    IMG_SIZE = 256
    SEED = 42
    LOSS = "Dice Loss"
    OPTIM = "adam"
    FILES = [
        "train_images_hflip", "train_images_vflip", "train_images_rotate_5", "train_images_rotate_10", "train_images_rotate_20", "train_images_rotate_45",
        "train_images_rotate_60", "train_images_rotate_90", "train_images_shear_x_0.1", "train_images_shear_x_0.25", "train_images_shear_x_0.5", 
        "train_images_shear_x_0.75", "train_images_shear_x_1", "train_images_shear_y_0.1", "train_images_shear_y_0.25", "train_images_shear_y_0.5", 
        "train_images_shear_y_0.75", "train_images_shear_y_1", "train_images_translate_25", "train_images_translate_50", "train_images_translate_75", "train_images_translate_100",
    ]

    for f in FILES:
        images_dir = "./images/" + f + "/"
        masks_dir = "./images/" + f.replace("_images_", "_masks_") + "/"
        aug_type = f[13:]
        main(IMG_SIZE, IMG_SIZE, OPTIM, LOSS, STARTING_LR, MIN_LR, BATCH_SIZE, 
             MAX_NUM_EPOCHS, f"./individual_aug/{aug_type}", images_dir, 
             masks_dir, "Trying out the effect of individual augmentations.", SEED)

def combined_augs():
    """
    Finds the best combination of individual augmentations.
    """
    
    STARTING_LR = 1e-4
    MIN_LR      = 1e-5
    BATCH_SIZE  = 4
    MAX_NUM_EPOCHS = 75
    IMG_SIZE = 256
    SEED = 42
    LOSS = "Dice Loss"
    OPTIM = "adam"
    
    FILES = [
        "train_images_rot10_shear_x1.0", "train_images_rot10_shear_x1.0_shear_y0.75", "train_images_rot10_shear_x1.0_shear_y0.75_trainslate50",
        "train_images_rot10_shear_x1.0_trainslate50", "train_images_rot10_shear_y0.75", "train_images_rot10_shear_y0.75_trainslate50", 
        "train_images_rot10_trainslate50", "train_images_rot10_hflip0", "train_images_rot10_hflip0_shear_x1.0", "train_images_rot10_hflip0_shear_x1.0_shear_y0.75", 
        "train_images_rot10_hflip0_shear_x1.0_shear_y0.75_trainslate50", "train_images_rot10_hflip0_shear_x1.0_trainslate50", 
        "train_images_rot10_hflip0_shear_y0.75", "train_images_rot10_hflip0_shear_y0.75_trainslate50", "train_images_rot10_hflip0_trainslate50", 
        "train_images_shear_x1.0_shear_y0.75", "train_images_shear_x1.0_shear_y0.75_trainslate50", "train_images_shear_x1.0_trainslate50", 
        "train_images_shear_y0.75_trainslate50", "train_images_hflip0_shear_x1.0", "train_images_hflip0_shear_x1.0_shear_y0.75", 
        "train_images_hflip0_shear_x1.0_shear_y0.75_trainslate50", "train_images_hflip0_shear_x1.0_trainslate50", "train_images_hflip0_shear_y0.75", 
        "train_images_hflip0_shear_y0.75_trainslate50", "train_images_hflip0_trainslate50"
    ]
    for f in FILES:
        images_dir = "./images/" + f + "/"
        masks_dir = "./images/" + f.replace("_images_", "_masks_") + "/"
        aug_type = f[13:]
        main(IMG_SIZE, IMG_SIZE, OPTIM, LOSS, STARTING_LR, MIN_LR, BATCH_SIZE, 
             MAX_NUM_EPOCHS, f"./combined_aug/{aug_type}", images_dir, 
             masks_dir, "Trying out the effect of combined augmentations.", SEED)


if __name__ == "__main__":
    #hyperparameter_tuning()
    #best_loss_and_optimizer()
    #find_good_seed()
    #individual_augs()
    #combined_augs()
    pass

