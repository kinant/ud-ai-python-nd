accuracy = 0.88  # use this as input for your submission

# Establish the default performance level to None
performance = None

# Use the accuracy value to assign performance levels to the correct performance names
if 0.91 <= accuracy <= 0.99:
    performance = "Excellent"
elif 0.76 <= accuracy <= 0.90:
    performance = "Good"
elif 0.51 <= accuracy <= 0.75:
    performance = "Average"
elif 0.00 <= accuracy <= 0.50:
    performance = "Poor"
else:
    performance = None

# Use the truth value of performance to assign result to the correct phrase
if performance:
    result = "The model has achieved {} performance.".format(performance)
else:
    result = "Performance level not defined."

### Notebook grading
if result == "The model has achieved Good performance.":
    print("Good work!")
else:
    print("Not quite! Are your result string formatted correctly?")