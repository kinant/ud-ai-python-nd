# PROGRAMMER: Kinan Turman
# DATE UPDATED: Oct. 22, 2025
# PURPOSE: Helper functions to use with PyTorch
import torch
from torch.utils.data import DataLoader

def get_best_cuda_device() -> torch.device:
    """
    Function to get the best cuda device
    Assumptions:    there is a CUDA device available
    Reasoning:      Can be used when a machine has more than one GPU available for example
                    As for example, my machine has an RTX4090 and an RTX5090
    Reference: # https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_device_capability.html
    :return: the best cuda device based on device major cuda capability
    """

    # Get the max index of the device with the best major cuda capability
    # We use the max function with a list of the device ids (as ints)
    # and the major device capability (index 0) as the key
    best_device = max(
        range(torch.cuda.device_count()),
        key=lambda i: torch.cuda.get_device_capability(i)[0]
    )
    # Return the best cuda device, based on major device capability
    return torch.device(f"cuda:{best_device}")

def get_device(override: str = None) -> torch.device:
    """
    Function to get the device to use by PyTorch
    :param override: Optional override for device
    :return: The best device available or the overridden device
    Reference: https://mctm.web.id/blog/2024/PyTorchGPUSelect/
    """
    # Check not overriding device to use
    if not override:
        # Step 1: Check if CUDA is available
        if torch.cuda.is_available():
            # If so, return the best CUDA device
            return get_best_cuda_device()
        # Step 2: If no CUDA, check if MPS enabled machine
        elif torch.backends.mps.is_available():
            # If so, return as device
            return torch.device("mps")
        # Step 3: If no CUDA and no MPS, then we return CPU
        else:
            return torch.device("cpu")
    else:
        # Else, if we are overriding, we specifically return
        # the device given by the argument (i.e. "cpu", "cuda", "cuda:0", "mps", etc)
        return torch.device(override)

def print_device_info(device: torch.device) -> None:
    """
    Function that prints device information
    :param device: the device to print information about
    """
    print(f"Using device: '{device}', with name: '{torch.cuda.get_device_name(device)}'")

def get_batch_data(dataloader: DataLoader) -> tuple:
    """
    Function to get a batch of data from a DataLoader
    :param dataloader: the dataloader to get data from
    :return: batch of images and labels as a tuple
    """
    if dataloader is not None:
        images, labels = next(iter(dataloader))
        return images, labels
    return None, None