import torch

def main():
    device = get_device()
    print(f"Using device: '{device}', Name: {torch.cuda.get_device_name(device)}")

def get_best_cuda_device():
    """
    Function to get the best cuda device
    Assumptions: there is a CUDA device available
    Reference: # https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_device_capability.html
    :return: the best cuda device based on device major cuda capability
    """
    best_device = max(
        range(torch.cuda.device_count()),
        key=lambda i: torch.cuda.get_device_capability(i)[0]
    )
    return torch.device(f"cuda:{best_device}")

def get_device(override=None):
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

if __name__ == '__main__':
    main()