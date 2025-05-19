model_accuracy_dict = {
    'ResNet': 0.91, 'MobileNet': 0.89, 'VGG': 0.88, 'Inception': 0.92,
    'AlexNet': 0.85, 'EfficientNet': 0.93, 'SqueezeNet': 0.87
}

print(model_accuracy_dict, '\n')

# find number of unique keys in the dictionary
num_models = len(model_accuracy_dict.keys())
print(num_models)

# find whether 'SqueezeNet' is a key in the dictionary
contains_squeezenet = 'SqueezeNet' in model_accuracy_dict.keys()
print(contains_squeezenet)

# create and sort a list of the dictionary's keys
sorted_keys = sorted(model_accuracy_dict.keys())
print(sorted_keys)

# get the first element in the sorted list of keys
first_key = sorted_keys[0]
print(first_key)

# find the element with the highest value in the dictionary
highest_accuracy_model = max(model_accuracy_dict, key=model_accuracy_dict.get)
print(highest_accuracy_model)

correct_num_models = 7
correct_contains_squeezenet = True
correct_first_key = 'AlexNet'
correct_highest_accuracy_model = 'EfficientNet'

if num_models != correct_num_models:
    print(f"Not quite! The number of unique models should be {correct_num_models}.")
elif contains_squeezenet != correct_contains_squeezenet:
    print(f"Not quite! Check if 'SqueezeNet' is a key in the dictionary.")
elif first_key != correct_first_key:
    print(f"Not quite! The first key when sorted should be {correct_first_key}.")
elif highest_accuracy_model != correct_highest_accuracy_model:
    print(f"Not quite! The model with the highest accuracy should be {correct_highest_accuracy_model}.")
else:
    print("Nice job! You can see my solution in the next page.")
