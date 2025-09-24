# Function to process data
def process_data(value):
    try:
        # Perform a division operation
        result = 100 / value
        print(f"Result: {result}")
    except ZeroDivisionError:
        # Handle the division by zero exception
        # TODO: write a friendly error message
        print("ERROR! Dividing by 0!")

# Test the function with an invalid value (0) to trigger the exception
process_data(0)