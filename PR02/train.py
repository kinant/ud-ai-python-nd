# PROGRAMMER: Kinan Turman
# DATE UPDATED: Oct. 22, 2025
# PURPOSE: Program that trains a pre-trained model on the flowers dataset
import argparse
from classifier import ImageClassifier
from dataloader import ImageDataLoader

CAT_TO_NAME_FILE = 'cat_to_name.json'

def get_command_line_args():
    parser = argparse.ArgumentParser()

    # Create the required command line arguments:
    # 1. directory to save checkpoints
    # 2. architecture
    # 3. hyperparameters: learning rate, hidden_units, epochs
    # 4. use gpu for training

    parser.add_argument('data_dir', type=str, default='flowers', help='data directory required')

    parser.add_argument('--save_dir', type=str, default='', help='path to save checkpoint')

    parser.add_argument('--arch', type=str, default=
    'alexnet', help='CNN Model Architecture to use')

    parser.add_argument('--learning_rate', type=float, default=
    0.001, help='training learning rate')

    parser.add_argument('--hidden_units', type=int, default=
    1024, help='hidden units for training')

    parser.add_argument('--epochs', type=int, default=
    10, help='number of epochs for training')

    # https://stackoverflow.com/questions/5262702/argparse-module-how-to-add-option-without-any-argument
    parser.add_argument('-g', '--gpu', action='store_true')

    return parser.parse_args()


def main():

    # Get the input args
    input_args = get_command_line_args()
    print()
    print('input_args', input_args)

    # Init variables
    data_dir = input_args.data_dir
    save_dir = input_args.save_dir

    arch = input_args.arch
    learning_rate = input_args.learning_rate
    hidden_units = input_args.hidden_units
    epochs = input_args.epochs
    use_gpu = input_args.gpu

    # STEP 1: LOADING DATA
    print()
    print(f"========== STEP 1: LOADING DATA ==========")
    dataloader = ImageDataLoader(data_dir)
    print(f"========== STEP 1: LOADING DATA COMPLETE ==========")
    print()

    # STEP 2: TRAIN THE MODEL/CLASSIFIER
    # Step 2a: Init the Image Classifier
    image_classifier = ImageClassifier(model_name=arch,
                                       lr=learning_rate,
                                       n_hidden=hidden_units,
                                       n_epochs=epochs,
                                       use_cuda=use_gpu,
                                       n_classes=dataloader.num_classes)
    image_classifier.init_model()

    # Step 2b: Set Classifier and Train
    print(f"========== STEP 2: TRAINING MODEL ==========")
    image_classifier.set_classifier()

    image_classifier.show_model_summary()
    image_classifier.show_device_info()

    print(f"--- BEGINNING TRAINING....")
    results = image_classifier.train(dataloaders=dataloader.dataloaders, num_epochs=epochs)
    print(f"========== STEP 2: TRAINING COMPLETE ==========")

    # STEP 3: SAVE CHECKPOINT
    print()
    print(f"========== STEP 3: SAVING CHECKPOINT ==========")
    image_classifier.save_checkpoint(save_dir, dataloader)
    print(f"========== STEP 3: SAVING  COMPLETE ==========")

    # STEP 4: TEST NETWORK
    print()
    print(f"========== STEP 4: TESTING NETWORK ==========")
    image_classifier.check_accuracy_on_test_data(dataloader=dataloader.dataloaders['test'])
    print(f"========== STEP 4: TESTING NETWORK COMPLETE ==========")

if __name__ == '__main__':
    main()