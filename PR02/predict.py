import torch
import dataloader
from PR02.helpers import plot_helpers as ph

# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#inference-on-custom-images
def predict(image_path, model, transform, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.to('cpu')

    processed_image = ph.process_image_simple(image_path, transform)
    processed_image = torch.unsqueeze(processed_image, 0)

    # processed_image.to("cuda")

    with torch.no_grad():

        logps = model.forward(processed_image)
        ps = torch.exp(logps)
        probs, classes = ps.topk(topk)

    return probs, classes
