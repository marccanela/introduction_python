"""
@author: mcanela
"""

"""
USING PYTHON AS A CALCULATOR

Assign to a Variable (=)
Add (+)
Subract (-)
Multiply (*)
Divide (/)
Power (**)
Integer Divide (//)
Remainder after Division (%)
"""

# What is two plus three?
2 + 3

# What is two times three?
2 * 3

# What is two to the third power?
2 ** 3

# How many (whole) times does 7 go into 100?
100 // 7

# What is the remainder after dividing 3 from 2?
3 % 2

"""
TESTING YOUR LOGIC: BOOLS

You can also ask Python logical questions (questions that have a "True" or "False" answer).
In this case, Python will return "True" if the statement is true, "False" if False.

Is Equal to (==)
Is Greater than (>)
Is Greater than or Equal to (>=)
Is not Equal to (!=)
"""

# One hundred is bigger than ten.
100 > 10

# Thirty squared is not equal to nine-hundred.
30 ** 2 != 900

# These two equations are equal: 30 - 5 * 2 and 100 / 5
30 - 5 * 2 == 100 / 5

# The sum of the numbers from 1 to 5 are at least 15.
1 + 2 + 3 + 4 + 5 >= 15
sum([1, 2, 3, 4, 5]) >= 15

# 0.2 plus 0.1 equals 0.3
0.2 + 0.1 == 0.3
0.2 + 0.1

"""
We do calculations using decimal (base 10), while computer does calculations using binary
(base 2). Decimals cannot be represented accurately in binary no matter how many significant 
digits you use.
"""

"""
PYTHON AS AN ECOSYSTEM OF FUNCTIONS

There are millions of functions that can be used in Python. To access them, you first have
to import their packages/modules.

>>> import package as abbreviation

Then you can specify the functions inside the packages using the following syntax:
    
>>> package.function(input)

Pyton has a reputation for being "batteries included", meaning it has a big standard library
of built-in packages!

To get a list of functions in a package, try using the dir() function.
To learn what the function does, try the help() function or put a question mark after
the function name.

>>> dir(math)
>>> help(math.sqrt)
>>> math.sqrt?
"""

# What is the square root of 16?
import math
math.sqrt(16)

# What is the log of 20?
math.log(20)

# What is pi?
math.pi

# What how many radians are in 360 degrees? (Functions tend to be named after their output)
help(math.radians)
math.radians(360)

# What is the cosine and sine of pi?
help(math.cos)
math.cos(math.pi)
math.sin(math.pi)

"""
AGGREGATING YOUR COLLECTIONS INTO SINGLE VALUES

Python can use named functions that turn data into something else. By doing this repeatedly,
in sequence, we can build data processing pipelines! Functions in Python have the following
syntax:
    
>>> function(input)

Some built-in functions are: min(), max(), sum(), and len().
"""

# The minimum of this list of numbers:
min([3, 6, 5, 2])

# The maximum of this list of numbers:
max([3, 6, 5, 2])

# The length of this list of numbers:
len([3, 4, 5, 5, 4])

# The sum of this list of numbers:
sum([1, 2, 5, 6])

# What should Python return when the code below is run?
max([min([1, 2]), max([3, 4, 5])])
min([max((2, 4, 1)), len((2, 4, 1)), min(2, 4, 1)])
len([max((2, 4, 1)), sum((2, 4, 1)), min(2, 4, 1)])
sum([max((2, 4, 1)), len((2, 4, 1)), min(2, 4, 1)])

"""
GROUPING TOGETHER DATA INTO A COLLECTION

Python also has operators for collecting related data together. There are pros and cons of each way of collecting data.

"tuple" (fixed sequence): (1, 2, 3)
"list" (changeable sequence): [1, 2, 3]
"str" (sequence of text characters): "123" or '123'
"set" {mathematical set): {1, 2, 3}
"""

# Make a list of your three favorite numbers
[1, 2, 5]

# Set x equal to your first name
x = "Marc"

# Make a tuple containing 3 names of people in your group.
("Irmak", "Ignasi", "Ana")

# Make a list of four animals, ordering them by size.
("mouse", "cat", "horse", "elephant")

# Collect the set of all letters in your whole name.
{"m", "a", "r", "c", "c", "a", "n", "e", "l", "a"}

"""
STATISTICS FUNCTIONS FROM NUMPY

Numpy is a Python package that, among other things, has many useful statistics functions.
Sometimes, the same functionality can be found both as a Numpy function and an array method,
giving you the choice of how you'd like to use it. Some useful function names: mean, min,
sum, max, var, std, p2p, median, nanmedian, nanmax, nanmean, nanmin

>>> import numpy as np
>>> np.mean([1, 2, 3, 4])


A couple lists of functions in Numpy can be found here:
Math: https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html
Statistics: https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.statistics.html
"""

import numpy as np
data = [2, 8, 5, 9, 2, 4, 6]

# What is the mean of the data?
np.mean(data)

# What is the sum of the data?
np.sum(data)

# What is the minimum of the data?
np.min(data)

# The variance?
var = np.var(data)
gdl = len(data) / (len(data) - 1)
gdl * var

# The standard deviation?
np.std(data)
gdl = len(data) / (len(data) - 1)
import math
math.sqrt(gdl * np.var(data))

"""
The functions np.var and np.std return the uncorrected values, while the same functions
in R programming language return the corrected values.
"""

# The difference between the data's maximum and minimum? ("peak-to-peak")
np.ptp(data)

# The data's median?
np.median(data)

# What about this data's median?
data2 = [2, 6, 7, np.nan, 9, 4, np.nan]
np.median(data2)
np.nanmedian(data2)

"""
np.nan is used for a missing value, and np.median returns nan if there is a np.nan.
np.nanmedian omits the np.nan values and gives a numeric value.
"""

"""
EXTRACTING DATA FROM A COLLECTION: INDEXING AND SLICING

Data can be indexed/queried/extracted from collections using the square brackets: [ ]
In sequences, putting a number inside the the brackets extracts the
nth (counting from zero) value

>>> (1, 2, 3)[1]
2
>>> (1, 2, 3)[0]
1
>>> (1, 2, 3)[-1]
3

You can "slice" a sequence (get all from one index to another index) using the colon [:]
The mathematical synthax behind is [...)
                                    
>>> (10, 20, 30, 40, 50, 60)[1:3]
(20, 30)
>>> (10, 20, 30, 40, 50, 60)[:3]
(10, 20, 30)
>>> (10, 20, 30, 40, 50, 60)[3:]
(40, 50, 60)
"""

data = (0.2, 0.3, 0.9, 1.1, 2.2, 2.9, 0.0, 0.7, 1.3, 0.3, 0.5, 0.1, 0.0)

# The first score
data[0]

# The third score
data[2]

# The last score
data[-1]

# The 3rd from the last score
data[-3]

# The 2nd through 5th score
data[1:5]

# Every second score (the first, third, fifth, etc)
data[::2]

# Every score after the 4th score (not included)
data[4:]

# Every second score from the 2nd to the 8th.
data[1:8:2]

# Every score except the first and last.
data[1:-1]

# Write it backwards
data[::-1]

'''
BUILDING ARRAYS

Numpy has a very useful list-like class: the array. It has one key restriction that lists don't have: 
all elements in the array have to be of the same data type (e.g. int, float, bool),
but it has a lot of useful features that lists also don't have, starting with having functions
that let you build arrays containing a wide range of starting data.

np.array()            Turns a list into an array. E.g. np.array([2, 5, 3])
np.arange()	          Makes an array with all the integers between two values. E.g. np.arange(2, 7)
np.linspace()	      Makes a specific-length array. E.g. np.linspace(2, 3, 10)
np.zeros()	          Makes an array of all zeros. E.g. np.zeros(5)
np.ones()	          Makes an array of all ones. E.g. np.ones(3)
np.random.random()	  Makes an array of random numbers. E.g. np.random.random(100)
np.random.randn()	  Makes an array of normally-distributed random numbers. E.g. np.random.randn(100)

Arrays are simple to work with and can crunch a lot of numbers in a short time!
On the other hand, if you want maximum flexibility (e.g. mix data types), lists are perfect!
'''

# Turn this list into an array: [4, 7, 6, 1]
import numpy as np
np.array([4, 7, 6, 1])

# Make an array containing the integers from 1 to 15.
np.arange(1, 16)

# Make an array of only 6 numbers between 1 and 10, evenly-spaced between them.
np.linspace(1, 10, 6)

# Turn this list into a an array: a = [True, False, False, True]
a = [True, False, False, True]
np.array(a)

# Make an array containing 20 zeros.
np.zeros(20)

# Make an array contain 20 twos!
np.ones(20) * 2

# How about an array of the 10 values between 100 and 1000?
np.linspace(100, 1000, 10)

# Generate an array of 10 random numbers
np.random.random(10)

'''
COMBINING AN ARRAY WITH STATISTICS FUNCTIONS
These exercises all involve two steps: (1) Make the data, and (2) Calculate something on the data
for example:

# the mean of the integers from 1 to 9
>>> np.mean(np.arange(1, 10))
'''

# What is the standard deviation of the integers between 2 and 20?
np.std(np.arange(2,21))

# What is the standard deviation of the numbers generated from the np.random.randn() function?
np.std(np.random.randn(1000000))

# What is the sum of an array of 100 ones?
np.ones(100).sum()

# What is the sum of an array of 100 zeros?
np.zeros(100).sum()

'''
PLOTTING DATA WITH THE MATHPLOTLIB PACKAGE

Sometimes data is straightforward to understand if simply printed as numbers, like we've been doing.
But visual charts are often much more effective at conveying patterns in data!
Matplotlib can create plots out of collections of data.
These figures can then be saved as image files and shared with others, like in presentations or papers.

>>> import matplotlib.pyplot as plt
>>> plt.plot([1, 2, 4, 9, 20])
>>> plt.savefig("myplot.png")

Special Things to Look Out For:
- Line Order Matters: Pay attention to what happens when code is written in a different order
  (e.g. data creation before plotting, plotting before labeling)
- Functions Have Optional Parameters that Change Their Behavior: Many of the plotting functions
  will let you modify them by adding a named parameter
  (e.g. plt.hist(data, bins=20, density=True, cumulative=False)).
  
plt.plot(data): Makes a line plot, with the y axis being the values in data
plt.plot(x, y): Makes a line plot, with the x and y axis matching the values in x and y.
'''

import numpy as np
import matplotlib.pyplot as plt

# Make a line plot of this list:
a = [1, 2, 4, 2, 1]
plt.plot(a)

# Make a line plot of this list:
b = np.sin(np.linspace(0, np.pi * 4, 200))
plt.plot(b)

# Notice that the above plot's x axis doesn't quite match the values.
# Make a line plot of the data, specifying both the x and y axis this time.
x = np.linspace(0, np.pi * 4, 200)
data = np.sin(x)
plt.plot(x, data);

'''
When adding ';' at the end of plt.plot() we just obtain the plot without the output in the console.
'''

'''
Let's add some text to our plots, so people understand what the plots represent!

plt.title("My Title"): Adds a title to the current plot
plt.xlabel("X"):  Adds a label to the x axis
plt.ylabel("Y"): Adds a label to the y axis
plt.plot(data, label="My label"); plt.legend():  Labels the plotted data and makes a legend with the labels.
Note: Normally this is done on seperate lines of code.

>>> plt.plot([1, 2, 4, 8, 16], label="Multiples of 2")
>>> plt.plot([1, 3, 9, 27], label="Multiples of 3")
>>> plt.title("Comparing Multiples of Integers")
>>> plt.xlabel("X")
>>> plt.ylabel("Y")
>>> plt.legend();
'''

# Make a line plot of the data, add the title "Sin of X" and label the Y axis "Sin of X":
x = np.linspace(0, np.pi * 10, 200)
data = np.sin(x)
plt.plot(x, data)
plt.title("Sin of X")
plt.ylabel("Sin of X");

# Make a line plot of each dataset put the labels "Sin of X" and "Cos of X" in a legend:
x = np.linspace(0, np.pi * 10, 200)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.plot(x, y_sin, label='Sin of X')
plt.plot(x, y_cos, label='Cos of X')
plt.legend();

'''
Histograms
When you have a lot of data, often you just want to know how much data you have and
around what values your data is clustered (the data's "distribution").
This is where histograms come in; on the x axis, they show the values in your data,
and on the y axis, how often those values (grouped together in "bins") occured in your dataset.

plt.hist(data): Make a histogram of the data
plt.hist(data, bins=20): Make a histogram of the data, specifying the number of bins
plt.hist(data, density=True): Make a probability distribution of the data (normalizes the y axis)
plt.hist(data, cumulative=True): Make a cumulative histogram of the data
plt.hist(data, label="My data"); plt.legend(): label the data and put the label in a legend.
plt.hist(data, alpha=0.5): Specify how opaque (i.e. how not-transparent) the plot is, from 0-1.
'''

# Make a histogram of this data.
# Looking at the plot, how would you describe the data's distribution (min, max, average, shape)?
n_data_points = 20
data = np.random.random(size=n_data_points)
plt.hist(data);

n_data_points = 200
data = np.random.random(size=n_data_points)
plt.hist(data);

n_data_points = 2000
data = np.random.random(size=n_data_points)
plt.hist(data);

n_data_points = 2000
data = np.random.normal(size=n_data_points)
plt.hist(data);

# Make a histogram of this data, setting the number of bins to 100, then 20, then 10, then 5
# How does the number of bins affect your interpreation of the data's distribution?
n_data_points = 2000
data = np.random.normal(size=n_data_points)
plt.hist(data, bins=100);

n_data_points = 2000
data = np.random.normal(size=n_data_points)
plt.hist(data, bins=20);

n_data_points = 2000
data = np.random.normal(size=n_data_points)
plt.hist(data, bins=10);

n_data_points = 2000
data = np.random.normal(size=n_data_points)
plt.hist(data, bins=5);

# Make a histogram of this data with normalized Y values (a probability density).
n_data_points = 200
data = np.random.normal(size=n_data_points)
plt.hist(data, density=True);

# Make a cumulative histogram of this data.
n_data_points = 200
data = np.random.normal(size=n_data_points)
plt.hist(data, cumulative=True, bins=200);

# Make two histograms, one for each of the two datasets.
# Label the data in a legend, and make the data more transparent for easier viewing.
n_data_points = 200
data1 = np.random.normal(0, 1, size=n_data_points)
data2 = np.random.normal(2, 0.5, size=n_data_points)
plt.hist(data1, alpha=0.5, label="Data1")
plt.hist(data2, alpha=0.5, label="Data2")
plt.legend();

'''
Other types of plots

Scatter Plots
>>> plt.scatter(x, y)

Heatmaps
>>> plt.imshow(data)
>>> plt.imshow(data, cmap='gray')
>>> plt.imshow(data, cmap='jet')
>>> plt.imshow(data, cmap='viridis') (This is the default one)
>>> plt.imshow(data, cmap='PiYG')

Subplots
>>> plt.subplot(no of rows, no of columns, active subplot)
'''

# Make a scatter plot of this data:
x = np.random.uniform(-5, 5, size=200)
noise = np.random.normal(0, 12, size=200)
y = x ** 3 + noise
plt.scatter(x, y);

# Make a heatmap of the 2D data:
data_1d = np.sin(np.linspace(-np.pi * 4, np.pi * 4, 200))
data_2d = np.tile(data_1d, (200, 1))
plt.imshow(data_2d);

# Make a sublpot of the following data:
plt.subplot(1, 2, 1)
plt.plot(x)
plt.title('CDF, More bins, More Labels')
plt.xlabel("X axis")
plt.ylabel('Y Axis')
plt.xlim(-3.5, 20);

plt.subplot(1, 2, 2)
plt.plot(x, label='My Data')
plt.plot(x*2, label='My Data*2')
plt.legend();

'''
MULTIDIMENSIONAL ARRAYS WITH NUMPY

Numpy arrays can be multidimensional: they can be squares, cubes, hypercubes, etc!
When choosing datastructures, Arrays are best chosen when all of the values in the structure
represent the same variable.

With multidimensional arrays, everything is pretty much the same as the 1-dimensional case,
with the addition of a few options for specifiying which order the dimensions should be in,
and which dimension an operation should operate on.

Most of the array-generation functions have a shape or size optional argument in them.
If you provide a tuple with a new shape specifying the number of elements along each dimension
(e.g. (5, 3) will produce a matrix with 5 rows and 3 columns), it will give you something multidimensional!

>>> data = np.random.randint(1, 10, size=(4, 5))
>>> data
array([[9, 7, 4, 2, 3],
       [3, 6, 7, 4, 8],
       [3, 6, 8, 7, 3],
       [6, 9, 4, 2, 2]])

For cases where there is no such option, all arrays have a reshape() method that lets you take
an existing array and make it more-dimensional. To simply flatten the matrix to a single dimension,
you can use the flatten() method.

>>> data.reshape(2, 10)
array([[9, 7, 4, 2, 3, 3, 6, 7, 4, 8],
       [3, 6, 8, 7, 3, 6, 9, 4, 2, 2]])

>>> data.flatten()
array([9, 7, 4, 2, 3, 3, 6, 7, 4, 8, 3, 6, 8, 7, 3, 6, 9, 4, 2, 2])

Numpy also has some auto-calculation features to make it a bit easier to get the shape you need:
if you don't know he nummber of rows/columns, just add '-1' and it'll be auto-calculated by Python.

>>> data.reshape(-1, 5)
array([[9, 7, 4, 2, 3],
       [3, 6, 7, 4, 8],
       [3, 6, 8, 7, 3],
       [6, 9, 4, 2, 2]])

>>> data.flatten()[np.newaxis, :]  # Makes a 1xN array
>>> data.flatten()[None, :]  # Also Makes a 1xN array
>>> data.flatten()[:, None]  # Makes an Nx1 array

And if an array has some extra dimensions you don't care about (like a 32x1x1 array,
and you just want a 32 array), you can use the squeeze() method to squeeze out those extra dimensions!

Finally, you can find out the shape of a matrix by getting its shape attribute.  And to get the total number of elements, check its size attribute.

>>> data.shape
(4, 5)

>>> data.size
20
'''

# Generate a 3 x 10 array of random integers between 1 and 4 using np.random.randint
np.random.randint(1, 5, size=(3, 10))

# Make a flat array with all the values between 0 and 11, and then reshape it into a 3 x 4 matrix
data = np.arange(0, 12)
data.reshape(3, 4)

# Reshape the previous array into a matrix with 3 columns...
data.reshape(-1, 3)

'''
Reordering Dimensions
There are many ways to transpose matrices:

array.T
array.transpose()
np.transpose(array)
array.swapaxes()
'''

# Try using each transpose method on the array x.
x = np.arange(6).reshape(2, 3)
x.T
x.transpose()
np.transpose(x)
x.swapaxes(0, 1)

'''
Aggregating Across Axes
Almost all of the Numpy functions have an axis option, which lets you limit the operation to just that axis.
Take into account: COLUMNS (axis = 0) and ROWS (axis = 1)

For example, to get the mean of all rows:

>>> array = np.arange(12).reshape(3, 4)
>>> array.mean(axis=1)
array([4., 5., 6., 7.])

And the mean of the columns:

>>> array.mean(axis=0)
array([1.5, 5.5, 9.5])

Notice that the number of dimensions goes down by default whenever you aggregate across the axis.
If you'd like to keep the dimensions the same, you can also use the keepdims=True option:

>>> array.mean(axis=0, keepdims=True)
array([[1.5],
       [5.5],
       [9.5]])
'''

np.random.seed(42)
data = np.random.randint(0, 10, size=(5, 3)) * [1, 10, 100]
data

# What is the mean of each column?
data.mean(axis=0, keepdims=True)

# What is the standard deviation of each row?
data.std(axis = 1, keepdims = True)

# What is the maximum of each column?
data.max(axis = 0, keepdims = True)

# What is the mean of each column's median?
np.median(data, axis = 0).mean()

'''
Indexing Exercises
Numpy arrays work the same way as other sequences, but they can have multiple dimensions
(rows, columns, etc) over which to index/slice the array.
'''

scores = np.arange(1, 49).reshape(6, 8)
scores

# The first score in the 2nd row:
scores[1,0]

# The third-through-fifth columns:
scores[:,2:5]

# The last score:
scores[-1, -1]

# The 2nd through 5th score, in the 6th column:
scores[1:5, 5]

# All the scores greater than 20:
scores[scores > 20]

# The rectangle inscribed by scores 19, 22, 35, and 38:
scores[2:5,2:6]

# The rectangle inscribed by scores 42, 44, 12, and 10:
scores[1:6, 1:4]

# Change the 3rd column to all 10s:
scores[:,2] = 10

# Change the last score to 999:
scores[-1,-1] = 999

# Change the 4th row to 0:
scores[3,:] = 0

'''
WORKING WITH IMAGES

Image data is stored as a 3D matrix, storing the brightness of each pixel along 3 coordinates:

- Which row the pixel is in  (between 0 and the height of the image)
- Which column the pixel is in (betweeen 0 and the width of the image)
- What color channel the pixel is in (red, green, blue, and sometimes alpha)

White pixels usually have the highest brightness values, and black pixels the darkest.

Let's load an image and visualize it onscreen using Matplotlib, a plotting library.
Working with images generally uses these 3 functions:

Function	    Purpose	                                         Example
plt.imread()	Loads a image from a filename	                 plt.imread("brian.png"
plt.imshow()	Plots a multidimensional array as an image	     plt.imshow(my_image_array)
plt.imsave()	Saves an array as an image on the computer	     plt.imsave("new_image.jpg", my_array)
'''

# Read and plot the "Cells" image in the images folder.
image = plt.imread("cells.jpg")
plt.imshow(image);

# Index only the first 50 rows of the image and plot it. (This is called "cropping" an image)
plt.imshow(image[:49, :]);

# Crop and Plot only the left cluster of cells.
plt.imshow(image[:, 10:175]);

# Crop and Plot only the right cluster of cells.
plt.imshow(image[20:85, 175:]);

# Crop and Plot only the tiny bright spot in the right cluster.
plt.imshow(image[51:58, 264:276]);

# Change that bright spot's brightness to white (255), and plot the whole image again.
image[51:58, 264:276] = 255
plt.imshow(image);

# Create a color map
plt.imshow(image[:, 10:175, 1], cmap="viridis");
plt.colorbar();

red = image[:, :, 0]
green = image[:, :, 1]
blue = image[:, :, 2]
plt.imshow(red);
plt.colorbar();

image[:, :175, 0] = image[:, :175, 1]
plt.imshow(image);
plt.colorbar();

'''
PANDAS DATAFRAMES

A DataFrame, simply put, is a Table of data. It is a structure that contains multiple rows,
each row containing the same labelled collection of data types. Because each row contains the same data,
DataFrames can also be thought of as a collection of same-length columns!

Pandas is a Python package that has a DataFrame class. Using either the DataFrame class constructor
or one of Pandas' many read_() functions, you can make your own DataFrame from a variety of sources:

- From a List of Dicts: Dicts are named collections. If you have many of the same dicts in a list,
the DataFrame constructor can convert it to a Dataframe:
    
>>> friends = [
    {'Name': "Nick", "Age": 31, "Height": 2.9, "Weight": 20},
    {'Name': "Jenn", "Age": 55, "Height": 1.2},
    {"Name": "Joe", "Height": 1.2, "Age": 25, },
]
>>> pd.DataFrame(friends)

- From a Dict of Lists:

>>> df = pd.DataFrame({
    'Name': ['Nick', 'Jenn', 'Joe'], 
    'Age': [31, 55, 25], 
    'Height': [2.9, 1.2, 1.2],
})

- From a List of Lists: if you have a collection of same-length sequences, you essentially have
a rectangular data structure already!  All that's needed is to add some column labels.

>>> friends = [
    ['Nick', 31, 2.9],
    ['Jenn', 55, 1.2],
    ['Joe',  25, 1.2],
]
>>> pd.DataFrame(friends, columns=["Name", "Age", "Height"])

- From an empty DataFrame: If you prefer, you can also add columns one at a time,
starting with an empty DataFrame:

>>> df = pd.DataFrame()
>>> df['Name'] = ['Nick', 'Jenn', 'Joe']
>>> df['Age'] = [31, 55, 25]
>>> df['Height'] = [2.9, 1.2, 1.2]
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Please recreate the table below as a Dataframe using one of the approaches detailed above:

# Year	Product	 Cost
# 2015	Apples	 0.35
# 2016	Apples	 0.45
# 2015	Bananas	 0.75
# 2016	Bananas	 1.10

df = pd.DataFrame()
df['Year'] = ["2015", "2016", "2015", "2016"]
df['Product'] = ["Apples", "Apples", "Bananas", "Bananas"]
df['Cost'] = [0.35, 0.45, 0.75, 1.1]
df

# Index the column 'Cost' and plot it. What's the mean of the costs?
df['Cost']
plt.plot(df['Cost']);
df['Cost'].mean()
df["Cost"].describe()

'''
You can easily create a summary of the descriptive statistics using describe()
'''
'''
READING DATA FROM FILES INTO A DATAFRAME

File Format                 File Extension          Read Function                           Dataframe Write Method
Comma-Seperated Values      .csv                    pd.read_csv()                           df.to_csv()
Tab-seperated Valuess       .tsv, .tabular, .csv    pd.read_csv(sep='\t'), pd.read_table()  df.to_csv(sep='\t') df.to_table()
Excel Spreadsheet           .xls                    pd.read_excel()                         df.to_excel()
Excel Spreadsheet 2010      .xlsx                   pd.read_excel(engine='openpyxl')        df.to_excel(engine='openpyxl')
JSON                        .json                   pd.read_json()                          df.to_json()
Tables in a Web Page (HTML) .html                   pd.read_html()[0]                       df.to_html()
HDF5                        .hdf5, .h5              pd.read_hdf5()                          df.to_hdf5()
'''

# Run the code below to download the Titanic passengers dataset, and transform it into different file formats
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
df = pd.read_csv(url)
df[:5]

# Save the dataframe to a TSV file.
df.to_csv("df.csv", sep='\t')

# Read the TSV file into Pandas again.
pd.read_csv("df.csv", sep='\t')

# Save the dataframe to an Excel file.
df.to_excel("file.xlsx", engine="openpyxl")
df.to_excel("newfile.xlsx", sheet_name="First")

'''
INDEXING ROWS AND COLUMNS OF A DATAFRAME

Because DataFrame rows are both ordered and named, they can be indexed using either approach,
and even both! On the other hand, column operations tend to be name-specific.
'''

# Select the "age" column
df['age']
df.age

# Get rows 10-16
df[10:16]

# Select the first 5 rows of the "sex" column
df[:5].sex

# Select the "embark_town" column
df.embark_town

# Select the "survived" and "age" columns:
df[['survived', 'age']]

# Select the last 3 rows of the "alive" column
df[-3:].alive

# Select rows 5-10 of the "class" column
df['class'][5:11]

# What is the mean ticket fare that the passengers paid on the titanic?
df.fare.mean()

# What is the median ticket fare that the passengers paid on the titanic?
np.median(df.fare)
df.fare.median()

# How many passengers does this dataset contain?
df.count()

# What class ticket did the 10th (index = 9) passenger in this dataset buy?
df['class'][9]

# What proportion of the passengers were alone on the titanic?
df.alone.value_counts(normalize=True)
df.alone.mean()

'''
Logicl values can be considered as numbers (True=1 & False=0), the mean is actually
the proportion of True values.
'''

# How many different classes were on the titanic?
df["class"].unique()

# How many men and women are in this dataset? (value_counts())
df.sex.value_counts()

# How many passengers are sitting in each class?
df['class'].value_counts()

# How many passengers of each sex are sitting in each class?
df[["sex", "class"]].value_counts()
df.groupby('class').sex.value_counts()

# Make a new column called "OnTitanic", with all of the values set to True
df['OnTitanic'] = True

# Make a new column called "isAdult", with True values if they were 18 or older and False if not.
df['isAdult'] = df.age >= 18

# Get everyone's age if they were still alive today (hint: Titanic sunk in 1912)
df['todayAge'] = df.age + (2022 - 1912)

# Make a column called "not_survived", the opposite of the "survived" column.
df['not_survived'] = df.survived == False

# Did the oldest passenger on the Titanic survive?
df.survived[df.age.max()]

# Where did the oldest passenger on the Titanic embark from?
df.embark_town[df.age.max()]

# How many passengers on the Titanic embarked from Cherbourg?
df.embark_town.value_counts()['Cherbourg']

'''
GROUPBY OPERATIONS

Usually, you don't just want to get a single metric from a dataset--you want to compare that metric
between differnt subgroups of your data. For example, you want the mean monthly temperature,
or the maximum firing rate of each neuron, and so on.

The groupby() method lets you specify that an operation will be done on each same-valued row
for a given column. For example, to ask for the mean temperature by month:

>>> df.groupby('month').temperature.mean()

To get the maxiumum firing rate of each neuron:

>>> df.groupby('neuron_id').firing_rate.max()

You can also group by as many columns as you like, getting as many groups as unique combinations
between the columns:

>>> df.groupby(['year', 'month']).temperature.mean()

Groupby objects are lazy, meaning they don't start calculating anything until they know the full pipeline.
This approach is called the "Split-Apply-Combine" workflow.
You can get more info on it here: https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
'''

# What was the mean age, grouped by class?
df.groupby('class').age.mean()

# What was the survival rate for each class?
df.groupby('class').survived.mean()

# What was the survival rate, broken down by both sex and class?
df.groupby(['sex', 'class']).survived.mean()

# Which class tended to travel alone more often?
# Did it matter where they were embarking from?
df.groupby(['class', 'embark_town']).alone.mean().unstack()

'''
unstack() is an interesting way to visualize data clearly.
'''

# What is mean ticket fare for the 1st class?
df.groupby('class').fare.mean()['First']

# And for the 1st class?
df.groupby('class').fare.mean()['Second']

# How many total people survived from Southampton?
df.groupby('embark_town').survived.value_counts()['Southampton'][1]

# From Cherbourg?
df.groupby('embark_town').survived.value_counts()['Cherbourg'][1]

# How many people from Southampton had first class tickets?
df.groupby('class').embark_town.value_counts()['First']['Southampton']

'''
PLOTTING WITH SEABORN

Seaborn is a data visualization library that uses Pandas Dataframes to produce
statitistical plots; in other words, it takes Dataframes and does Groupby automatically for you
'''

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# What was the average age of the people in each class?
sns.catplot(data=df, x="class", y="age", kind="swarm");
sns.catplot(data=df, x="class", y="age", kind="violin");

# What was the average survival rate, broken down by both sex and class?
sns.catplot(data=df, x="class", y="survived", hue="sex", kind="bar");

# What was the average age of the people, broken down by class, sex and embarking port?
sns.catplot(data=df, x="class", y="age", hue="sex", col="embarked", kind='swarm');

# What was the distribution of prices?
sns.distplot(df.fare, bins=40);

# And the correlation of prices according to age?
sns.jointplot(data=df, x="fare", y='age');
sns.jointplot(data=df, x="fare", y='age', kind="hex");

# And according to class?
sns.jointplot(data=df, x="class", y='fare');

'''
FILTERING DATA WITH LOGICAL INDEXING

Sometimes you want to remove certain values from your dataset. In Numpy, this can be done
with Logical Indexing, and in normal Python this is done with an If Statement

*Step 1: Create a Logical Numpy Array
We can convert all of the values in an array at once with a single logical expression.
This is broadcasting, the same as is done with the math operations we saw earlier:

>>> data = np.array([1, 2, 3, 4, 5])
>>> data < 3
[True, True, False, False, False]

*Step 2: Filter with Logical Indexing
If an array of True/False values is used to index another array, and both arrays are the same size,
it will return all of the values that correspond to the True values of the indexing array:

>>> data = np.array([1, 2, 3, 4, 5])
>>> is_big = data > 3
>>> is_big
[False, False, False, True, True]

>>> data[is_big]
[4, 5]

Both steps can be done in a single expression. Sometimes this can make things clearer!
'''

import numpy as np
list_of_values = [3, 7, 10, 2, 1, 7, np.nan, 20, -5]
data = np.array(list_of_values)

# Which values are greater than zero?
data > 0

# Which values are equal to 7?
data == 7

# Which values are greater or equal to 7?
data >= 7

# Which values are not equal to 7?
data != 7

# Using the data below, the values that are less than 0
data = np.array([3, 1, -6, 8, 20, 2, np.nan, 7, 1, np.nan, 9, 7, 7, -7])
data[data < 0]

# The values that are greater than 3
data[data > 3]

# The values not equal to 7
data[data != 7]

# The values equal to 20
data[data == 20]

# The values that are not missing
data[np.isfinite(data)]
data[~np.isnan(data)]

# Using the following dataset, How many values are greater than 4 in this dataset?
data = np.array([3, 1, -6, 8, 20, 2, 7, 1, 9, 7, 7, -7])
len(data[data > 4])

# How many values are equal to 7 in this dataset?
len(data[data == 7])

# What is the mean value of the positive numbers in this dataset?
data[data > 0].mean()

# What is the mean value of the negative numbers in this dataset?
data[data < 0].mean()

# What is the median value of the values in this dataset that are greater than 5?
np.median(data[data > 5])

# How many missing values are in this dataset?
len(data[np.isnan(data)])

# What proportion of the values in this dataset are positive?
len(data[data > 0]) / len(data)

newdata = data > 0
newdata.mean()

np.mean(data > 0)

# What proportion of the values in this dataset are less than or equal to 8?
np.mean(data <= 8)

'''
STATISICS WITH THE PINGOUIN PACKAGE

Pingouin is a new statistics package in Python that uses pandas, seaborn, and scipy-stats!
It's quite user-friendly and its documentation is good; let's use it to analyze some data!

https://pingouin-stats.org/index.html
'''

import statsmodels.api as sm
import pingouin as pg


# What significant differences are there between the fertility rates in 1990, 2000, and 2010?
# Parametric Tests: Follow the flowchart in the ANOVA section of the penguoin docs to test for differences
# in the mean fertility rate between these 3 years. Even the deta is not homoscedastic, go ahead and do the
# anova and pairwise tests
# https://pingouin-stats.org/guidelines.html#id5
pg.homoscedasticity(data=df, dv='1990', group='Country name')


a = pg.read_dataset('anova')























