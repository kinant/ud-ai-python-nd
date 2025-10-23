# PROGRAMMER: Kinan Turman
# DATE UPDATED: Oct. 22, 2025
# PURPOSE: Program that predicts the class of an image using a saved model
import torch
import argparse

from dataloader import ImageDataLoader
from classifier import load_checkpoint

from helpers import visualization as ph
import helpers.torch_helpers as thlp

def main():
    # Get the command line args
    input_args = get_command_line_args()

    img_path = input_args.img_path
    checkpoint_path = input_args.checkpoint
    top_k = input_args.top_k
    cat_names_file = input_args.category_names
    use_gpu = input_args.gpu

    print(f"{input_args}")

    # STEP 1: Load Checkpoint
    image_classifier = load_checkpoint(checkpoint_path)
    device = thlp.get_device() if use_gpu else thlp.get_device("cpu")

    # print("device: ", device)
    # image_classifier.show_model_summary()

    # STEP 2: Make the prediction
    probs, classes = predict(img_path, image_classifier.model, device, top_k)

    # print(f"Probs: {probs}\nClasses: {classes}")
    # print(f"Cat names: {cat_names_file}")

    # STEP:3: print the results
    if cat_names_file:
        print_results(classes, probs, cat_names_file, using_mapping=True)
    else:
        print_results(classes, probs, "", )

def get_command_line_args():
    # Create Parse using Argument Parser
    parser = argparse.ArgumentParser()

    # Create the required command line arguments:
    # 1. path to image
    # 2. checkpoint location
    # 3. top K most likely classes
    # 4. mapping of categories to real names
    # 5. use gpu for inference

    parser.add_argument('img_path', type=str, help='image path required')

    parser.add_argument('checkpoint', type=str, help='checkpoint path required')

    parser.add_argument('--top_k', type=int, default=
    1, help='Top K most likely classes')

    parser.add_argument('--category_names', type=str, default=
    None, help='File for mapping of categories to real names')

    parser.add_argument('-g', '--gpu', action='store_true')

    return parser.parse_args()

def predict(image_path, model, device, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    Reference: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#inference-on-custom-images
    """
    model.eval()
    model.to(device)

    # get the processed image and unsqueeze it (1-D vector)
    processed_image = ph.process_image(image_path)
    processed_image = torch.unsqueeze(processed_image, 0).to(device)

    with torch.no_grad():

        # Perform the forward step (inference) and get the probability logits
        logps = model.forward(processed_image)

        # convert to probabilities
        ps = torch.exp(logps)

        # get the top k probabilities and classes and return
        probs, classes = ps.topk(topk)

    return probs, classes

def print_results(classes, probs, filename, using_mapping=False):
    """
    Function that prints the results of predict/inference
    """
    print()
    print(f"========== PREDICTION RESULTS ==========")
    print(f"========================================")
    print()

    # check if using the mapping
    if using_mapping:
        dataloader = ImageDataLoader("", load_data=False)
        dataloader.load_cat_to_name(filename)

    # iterate over each class and probability and print the results
    for cls, p in zip(classes.tolist()[0], probs.tolist()[0]):
        if using_mapping:
            print(f"Class: '{dataloader.cat_to_name[str(cls)]}'", end="")
        else:
            print(f"Class: '{cls}'", end="")

        print(f", with probability: {p * 100:.2f}%")

    print()
    print(f"========================================")

if __name__ == '__main__':
    main()