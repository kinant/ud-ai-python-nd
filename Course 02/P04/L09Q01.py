# Define the sentence
sentence = "the quick brown fox jumped over the lazy dog"

# Tokenize the sentence into words
words = sentence.split()

# Print each word on a new line
for word in words:
    print(word)

# Define the dataset
data = list(range(1, 31))

# Define the batch size
batch_size = 5

# Process the data in batches
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    print("Batch # {}: {}".format(i // batch_size + 1, batch))