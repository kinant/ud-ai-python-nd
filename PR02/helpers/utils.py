# -------------------------------
# General Utilities
# -------------------------------
import importlib.util
import random
from os import path, listdir

# FOR NORMALIZING IMAGES
NORMALIZING_MEAN = [0.485, 0.456, 0.406]
NORMALIZING_STD = [0.229, 0.224, 0.225]

def is_pkg_installed(package: str):
    """
    Function to check if a package is installed
    Reference used: https://www.geeksforgeeks.org/python/how-to-check-if-python-package-is-installed/
    :param package: name of the package to check
    :return: True if the package is installed, False otherwise
    """
    return importlib.util.find_spec(package) is not None

def get_random_image_path():
    """
    Get a random image path from the flowers' dataset.
    I will use this function to
    use the predict function on any random image
    in that folder.
    """

    # get a random index for the folder (there are 102)
    rand_folder_idx = random.randint(1, 102)
    # print(rand_folder_idx)

    # get a random image
    # build the dir/path
    rand_dir = 'flowers/test/' + str(rand_folder_idx) + '/'

    # get a random file from the dir
    # https://stackoverflow.com/questions/701402/best-way-to-choose-a-random-file-from-a-directory
    rand_img = random.choice(listdir(rand_dir))

    # build fluid image path
    rand_img_path = rand_dir + rand_img

    return rand_img_path