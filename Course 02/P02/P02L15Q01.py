## String method playground

# Example usage of some string methods
sample_text = "Hello, World! This is a sample text for NLP tasks."

# Convert the text to lowercase
lower_text = sample_text.lower()
print("Lowercase:", lower_text)

# Replace 'World' with 'Universe'
replaced_text = sample_text.replace("World", "Universe")
print("Replaced text:", replaced_text)

# Split the text into words
words = sample_text.split()
print("Words:", words)

# Try out more string methods from the Python documentation link provided

# Write two lines of code below, each assigning a value to a variable
model_name = "BERT"
accuracy = 92.5

# Now write a print statement using .format() to print out a sentence and the values of both of the variables
print("Model {} has an accuracy of {:.2f}%".format(model_name, accuracy))