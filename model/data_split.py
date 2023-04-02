import os
import shutil
import random

# Set the path to the directory containing your data
data_dir = '../input'

# Set the path to the directory where you want to store your training data
train_dir = './train'

# Set the path to the directory where you want to store your testing data
test_dir = './test'

# Set the size of your testing data (as a percentage of the total data)
test_size = 0.2

# Loop through each emotion folder in your data directory
for emotion_folder in os.listdir(data_dir):
    # Create a subdirectory in the training and testing directories for this emotion
    os.makedirs(os.path.join(train_dir, emotion_folder), exist_ok=True)
    os.makedirs(os.path.join(test_dir, emotion_folder), exist_ok=True)
    
    # Get the list of image filenames in this emotion folder
    image_filenames = os.listdir(os.path.join(data_dir, emotion_folder))
    
    # Randomly shuffle the list of image filenames
    random.shuffle(image_filenames)
    
    # Split the list of image filenames into training and testing sets
    test_image_filenames = image_filenames[:int(len(image_filenames)*test_size)]
    train_image_filenames = image_filenames[int(len(image_filenames)*test_size):]
    
    # Move the training images to the training directory
    for filename in train_image_filenames:
        src_path = os.path.join(data_dir, emotion_folder, filename)
        dst_path = os.path.join(train_dir, emotion_folder, filename)
        shutil.copy(src_path, dst_path)
    
    # Move the testing images to the testing directory
    for filename in test_image_filenames:
        src_path = os.path.join(data_dir, emotion_folder, filename)
        dst_path = os.path.join(test_dir, emotion_folder, filename)
        shutil.copy(src_path, dst_path)
