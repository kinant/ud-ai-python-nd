def create_scientist_list(filename):
    """Reads the file and extracts a list of AI scientist names."""
    scientist_list = []
    #use with to open the file filename
    #use the for loop syntax to process each line
    #and add the scientist name to scientist_list
    try:
        with open(filename, 'r') as f:
            for line in f:
                scientist_list.append(line.split(',')[0])
    except FileNotFoundError:
        print(f"File {filename} not found")
    return scientist_list

# Create scientist list from the file
scientist_list = create_scientist_list('files/ai_scientists.txt')

# The correct result list for grading
correct_result = [
    'Alan Turing', 'Barbara Grosz', 'Cynthia Dwork', 'Daphne Koller', 'Erik Brynjolfsson', 'Fei-Fei Li',
    'Geoffrey Hinton', 'Hilary Mason', 'Ian Goodfellow', 'Judea Pearl', 'Kunihiko Fukushima', 'Leslie Valiant',
    'Marvin Minsky', 'Nando de Freitas', 'Oren Etzioni', 'Peter Norvig', 'Qiang Yang', 'Rodney Brooks',
    'Stuart Russell', 'Tim Berners-Lee', 'Ursula Martin', 'Vladimir Vapnik', 'Wendy Hall', 'Xiaojin Zhu',
    'Yann LeCun', 'Zoubin Ghahramani'
]

# Notebook grading
if scientist_list == correct_result:
    print("Well done!")
else:
    print("Your code produced the wrong result when running on the `ai_scientists.txt`.")
