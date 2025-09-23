#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Kinan Turman
# DATE CREATED: 09/23/2025
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

from torch.utils.hipify.hipify_python import value


def sanitize_text(text):
    """
    Simple function that takes a string and returns it sanitized, that is, without numbers, symbols, etc.
    :param text: the string to be sanitized
    :return: the sanitized string
    """

    # "".join uses "" (empty string) as the delimiter and joins all elements of the list to create a complete string
    return "".join(
        # using list comprehensions, we get a list of all characters, in lowercase, which are alphabetic or a space
        # from the stripped and split text (with spaces replacing underscores)
        [
            # lowercase the character
            c.lower()
            # for each character in the text (stripped, split - to remove the file extensions, and with '_'
            # replaced by ' ' a single space).
            for c in text.strip().split('.')[0].replace('_', ' ')
            # checking that it is alphabetic or a space character
            if c.isalpha() or c == ' '
        ]
    )

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    # Replace None with the results_dic dictionary that you created with this
    # function

    # get the list of filenames
    filenames = listdir(image_dir)

    # init blank dictionary
    results_dic = {}

    # iterate over each filename and add to the dictionary
    # key: filename
    # value: "sanitized" filename, as a list
    for name in filenames:
        # check not already in dictionary
        if name not in results_dic:
            # if not, add it
            results_dic[name] = [sanitize_text(name).strip()]
        else:
            # else print the warning
            print(f"Key {name} already exists in results_dic, with value {results_dic[name]}")

    return results_dic
