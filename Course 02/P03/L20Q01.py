model_descriptions = "ResNet is a convolutional neural network that is 50 layers deep. MobileNet is a lightweight convolutional neural network designed for mobile and embedded vision applications. VGG is a convolutional neural network model proposed by Visual Geometry Group of Oxford University. Inception is a deep convolutional neural network architecture that has achieved state-of-the-art results."

print(model_descriptions, '\n')

# TODO: replace None with appropriate code
# split model_descriptions into list of words
model_list = model_descriptions.split()
print(model_list, '\n')

# TODO: replace None with appropriate code
# convert list to a data structure that stores unique elements
model_set = set(model_list)
print(model_set, '\n')

# TODO: replace None with appropriate code
# find the number of unique words
num_unique = len(model_set)
print(num_unique, '\n')

### Notebook grading
correct_answer = 36  # Adjust this according to the actual count
if type(model_list) != list:
    print("`model_list` should be a list of all words in `model_descriptions`.")
elif type(model_set) != set:
    print("`model_set` should be a set of all unique words in `model_list`.")
elif type(num_unique) != int:
    print("Make sure you define `num_unique` with the number of unique words!")
elif num_unique != correct_answer:
    print("Not quite! Are you finding the length of the set correctly?")
else:
    print("Nice job! You can see my solution in the next page.")
