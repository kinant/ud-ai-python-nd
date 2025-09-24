accuracy = 0.85
result = ''

if 0.91 <= accuracy <= 0.99:
    result = "Model performance: Excellent."
elif 0.76 <= accuracy <= 0.90:
    result = "Model performance: Good."
elif 0.51 <= accuracy <= 0.75:
    result = "Model performance: Average."
elif 0.00 <= accuracy <= 0.50:
    result = "Model performance: Poor."
else:
    result = "Invalid accuracy score!"

# Check the result
print(result)

# Notebook grading
if result == "Model performance: Good.":
    print("Nice work!")
else:
    print("Not quite! Are your result strings formatted correctly?")