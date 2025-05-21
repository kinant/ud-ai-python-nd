model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"]
model_identifiers = []

# Write your for loop here
for model in model_names:
    model_identifiers.append("_".join(model.split()).lower())

print(model_identifiers)

### Notebook grading
if model_identifiers == ["logistic_regression", "decision_tree", "random_forest", "support_vector_machine"]:
    print("Nice work!")
else:
    print("Not quite! Did you append each new identifier to the list `model_identifiers`?")


model_identifiers = ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine"]

# Write your for loop here
for i in range(0, len(model_identifiers)):
    model_identifiers[i] = model_identifiers[i].lower().replace(" ", "_")

### Notebook grading
if model_identifiers == ["logistic_regression", "decision_tree", "random_forest", "support_vector_machine"]:
    print("Nice work!")
else:
    print("Not quite! Did you modify each element in the list `model_identifiers`?")

predictions = ['Predicted: 0.95', 'Actual: 0.90', 'Predicted: 0.85']
count = 0

# Write your for loop here
for prediction in predictions:
    if prediction.startswith("Predicted:"):
        count += 1

### Notebook grading
if count == 2:
    print("Nice work!")
else:
    print("Not quite! Did you track the number of predictions with `count`?")

metrics = ['Accuracy: 0.95', 'Precision: 0.92', 'Recall: 0.88']
html_str = "<ul>\n"

# Write your code here
for metric in metrics:
    html_str += "<li>{}</li>\n".format(metric)

html_str += "</ul>"

# Print the resulting HTML string
print(html_str)

print(list(range(0, -5)))