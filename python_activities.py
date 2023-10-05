"""
Created on Mon Nov 21 08:41:22 2022
@author: mcanela
"""

# EXERCISE 1

measure = 27.35
average = 21
experiment = 'subject-1_protocol-3'

# What is measure type?
# A stored variable, a float

# Write a line to test where measure is bigger than average and store the result in a variable called bigger.
bigger = measure > average

# Extract the protocol number from experiment
experiment[19]

# Extract the subject number from experiment
experiment[8]

# Extract both protocol and subject numbers using the split() method from strings.
splited = experiment.split('_')

subject = splited[0]
subject[8]

protocol = splited[1]
protocol[9]

# ====================================================================

subjects = ["mouse_1", "mouse_2", "mouse_3", "mouse_4", "mouse_5", "mouse_6"]

# Extract every other animal name ("mouse_1", "mouse_3", "mouse_5")
subjects[::2]

# Create a list with every animal number ([1, 2, 3, 4, 5, 6])
number = [int(mouse[6]) for mouse in subjects]

# ====================================================================

filenames = ['f2.csv', 'f1.csv', 'notes.txt', 'f3.csv']
animals = ["mouse_3", "mouse_2", "mouse_1"]

# Create a list with only csv files
del filenames[2]

# Create a dictionary associating files and animals, with the same number. Expected output:
# {'mouse_3': 'f3.csv', 'mouse_2': 'f2.csv', 'mouse_1': 'f1.csv'}

animals.sort(reverse = True)
filenames.sort(reverse = True)
dictionary = dict(zip(animals, filenames))

# ====================================================================

# EXERCISE 2

# Write an operation function that accepts two numbers and an operation as a string ("+", "-", "*", "/")
# and performs the corresponding operation, returning the result. Function should be usable like this:
    # operation(3, 4, "+") # Returns 7
    # operation(3, 4) # Returns 7
    # operation(3, 4, "-") # Returns -1
    # operation(3, 4, "*") # Returns 12
    # operation(3, 4, "/") # Returns 0.75


def operation(number1, number2, symbol = '+'):
    if symbol == '+':
        print(number1 + number2)
    if symbol == '-':
        print(number1 - number2)    
    if symbol == '*':
        print(number1 * number2)    
    if symbol == '/':
        print(number1 / number2)

# ====================================================================

# EXERCISE 3

# You are given a DNA sequence. Your job is to write a function named kmer with two parameters
# (seq, k): seq is the DNA sequence and k is an integer. The function should return a dictionary
# with the keys being all the subsequences of size k and the corresponding value being the
# number of times it occurs in seq.
    # seq = 'GAAGTAATGAA'
    # kmer(seq, 2) # --> {'GA': 2, 'AA': 3, 'AG': 1, 'GT': 1, 'TA': 1, 'AT': 1, 'TG': 1}
    # kmer(seq, 3) # --> {'GAA': 2, 'AAG': 1, 'AGT': 1, 'GTA': 1, 'TAA': 1, 'AAT': 1, 'ATG': 1, 'TGA': 1}

def kmer(seq, k):
    dictionary = {}
    letters = []
    for number in range(len(seq)):
        entry = seq[number:number + k]
        if len(entry) == k: 
            letters.append(entry)
    for letter in letters:
        count = letters.count(letter)
        dictionary[letter] = count
    print(dictionary)
    

# ====================================================================

# EXERCISE 4

# Write a function mov_average accepting as parameters: a list of numbers
# (input_values), an integer (win_size), with a default value of 3. This
# function has to return a list of numbers corresponding to the moving average
# of input_values, on a window of size win_size.
    # input_values = [11, 13, 1, 24, 1, 18, 14, 20, 10, 17, 9, 15]
    # win_size = 3
    # mov_average(input_values, win_size)
    # >> [8.333333333333334, 12.666666666666666,8.666666666666666, 14.333333333333334, 11.0,
    # 17.333333333333332, 14.666666666666666,  15.666666666666666, 12.0, 13.666666666666666]

def mov_average(input_values, win_size = 3):
    averages = []
    for number in range(len(input_values)):
        temporal_list = input_values[number:number + win_size]
        if len(temporal_list) == win_size: 
            mean = sum(temporal_list) / len(temporal_list)
            averages.append(mean)
    print(averages)
            

# ====================================================================

# EXERCISE 5

# Write a function max_pos taking as parameters a list of numbers and that
# return the index / indices of the maximum.
    # input_values = [11, 13, 1, 24, 1, 18, 14, 20, 10, 17, 9, 15]
    # max_pos(input_values)
    # >>> [3]
    # input_values = [11, 13, 1, 24, 1, 18, 14, 20, 10, 17, 9, 15, 24]
    # max_pos(input_values)
    # >>> [3, 12]
    
def max_pos(input_values):
    maximum = max(input_values)
    indices = []
    copy_input_values = input_values[:]
    while maximum in copy_input_values:
        entry = input_values.index(maximum)
        indices.append(entry)
        copy_input_values.remove(input_values[entry])
        input_values[entry] = input_values[entry] + 1
    print(indices)


# ====================================================================

# EXERCISE 6

# Write a function find_peaks taking as parameters: a list of numbers signal
# and an integer win_size, with a default value of 3. This function returns a
# list of indices peaks_pos corresponding to peaks in the signal. A point is
# considered a peak when it is greater or equal to all its neighbors in a window
# of size win_size * 2 (so with win_size - 1 neighbors on each side of itself).
# Note : Peaks on the edges (less than win_size from the beginning or the end) can be ignored.
    # signal = [0, 1, 0, 1, 12, 24, 19, 25, 12, 5, 0, 1, 1, 0, 14, 29, 28, 16, 6, 1, 0]
    # winsize = 3
    # find_peaks(signal)
    # >>> [7, 15]
    # find_peaks(signal, 2)
    # >>> [1, 5, 7, 11, 15]

def find_peaks(signal, win_size=3):
    indices = []
    size = 2 * win_size - 1
    middleIndex = int((size - 1)/2)
    for number in range(len(signal) - size + 1):
        temporal_list_prev = signal[number:number + middleIndex]
        temporal_list_post = signal[number + middleIndex + 1:number + middleIndex * 2 + 1]
        
        if not any(x > signal[number + middleIndex] for x in temporal_list_prev):
            prev = True
        else:
            prev = False
        if not any(x > signal[number + middleIndex] for x in temporal_list_post):
            post = True
        else:
            post = False
        if prev == True and post == True:
            indices.append(number + middleIndex)
    
    for x in range(len(indices) - 2):
        if indices[x] + 1 == indices[x + 1]:
            del indices[x + 1]
    
    print(indices)

















