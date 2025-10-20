# -------------------------------
# General Utilities
# -------------------------------
import importlib.util

# FOR NORMALIZING IMAGES
NORMALIZING_MEAN = [0.485, 0.456, 0.406]
NORMALIZING_STD = [0.229, 0.224, 0.225]

# https://www.geeksforgeeks.org/python/how-to-check-if-python-package-is-installed/
def is_pkg_installed(package: str):
    """
    Function to check if a package is installed
    :param package: name of the package to check
    :return: True if the package is installed, False otherwise
    """
    return importlib.util.find_spec(package) is not None