#!/usr/bin/env python
# coding: utf-8

# # Python libraries and an introduction to data manipulation
# 
# The core Python language is by design somewhat minimal.  Like other programming languages, Python has an ecosystem of modules (libraries of code) that augument the base language.  Some of these libraries are "standard", meaning that they are included with your Python distribution.  Many other open source libraries can be obtained from the organizations that support their development.
# 
# Think of a library as a collection of functions and data types that can be accessed to complete certain programming tasks without having to implement everything yourself from scratch.
# 
# This course will make extensive use of the following libraries:
# 
# * **[Numpy](http://numpy.org)** is a library for working with arrays of data.
# 
# * **[Pandas](http://pandas.pydata.org)** provides high-performance, easy-to-use data structures and data analysis tools.
# 
# * **[Scipy](http://scipy.org)** is a library of techniques for numerical and scientific computing.
# 
# * **[Matplotlib](http://matplotlib.org)** is a library for making graphs.
# 
# * **[Seaborn](http://seaborn.pydata.org)** is a higher-level interface to Matplotlib that can be used to simplify many graphing tasks.
# 
# * **[Statsmodels](http://www.statsmodels.org)** is a library that implements many statistical techniques.
# 
# This notebook introduces the Pandas and Numpy libraries, which are used to manipulate datasets.  Next week we will give an overview of the Matplotlib and Seaborn libraries that are used to produce graphs.  The Statsmodels package will be used in the second and third courses of the series that introduce formal statistical analysis and modeling. 
# 
# # Documentation
# 
# No data scientist or software engineer memorizes every feature of every software tool that they utilize.  Effective data scientists take advantage of resources (mostly on-line) to resolve challenges that they encounter when developing code and analyzing data.  Documentation is the official, authoritative resource for any programming language or library. Here are links to the official documentation for the [Python language](https://docs.python.org/3/) and the [Python Standard Library](https://docs.python.org/3/library/index.html#library-index).

# ### Importing libraries
# 
# When using Python, you will generally begin your scripts by importing the libraries that you will be using. 
# 
# The following statements import the Numpy and Pandas libraries, giving them abbreviated names:

# In[2]:


import numpy as np
import pandas as pd


# ### Utilizing library functions
# 
# After importing a library, its functions can then be called from your code by prepending the library name to the function name.  For example, to use the '`dot`' function from the '`numpy`' library, you would enter '`numpy.dot`'.  To avoid repeatedly having to type the libary name in your scripts, it is conventional to define a two or three letter abbreviation for each library, e.g. '`numpy`' is usually abbreviated as '`np`'.  This allows us to use '`np.dot`' instead of '`numpy.dot`'.  Similarly, the Pandas library is typically abbreviated as '`pd`'.

# The next cell shows how to call functions from an imported library:

# In[3]:


a = np.array([0,1,2,3,4,5,6,7,8,9,10]) 
np.mean(a)


# As you can see, we first used the `array` function from the numpy library to create a literal 1-dimensional array, and then used the `mean` function from the library to calculate its average value (this is called a "literal" array because the data are entered directly into the notebook).

# ## NumPy
# 
# NumPy is a fundamental package for scientific computing with Python. It includes data types for vectors, matrices, and higher-order arrays (tensors), and many commonly-used mathematical functions such as logarithms.
# 
# #### Numpy Arrays (the ndarray)
# 
# We are mainly interested in the [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) object which is an n-dimensional array of values, and the methods that allow us to manipulate such arrays.  Recall that a Python list may contain values of different types, e.g. [1, "pig", [3.2, 4.5]] is a Python list containing three elements -- an integer, a string, and another list that itself contains two floating point values.  Lists containing inhomogeneous types are convenient, but do not perform well for large-scale numerical computing.  The numpy ndarray is a homogeneous array that may have any number of axes.  Since it is homogeneous, all values in one ndarray must have the same data type (e.g. all values are integers, or all are floating point numbers).
# 
# A numpy array is a table of values that may have any number of "axes".  A 1-dimensional numpy array has a single axis, and is somewhat analogous to a Python list or a mathematical vector.  A 2-dimensional numpy array has two axes, and can be seen as a table or matrix.  Higher-order arrays (tensors) can be useful in specific cases, but are not encountered as often.  As noted above, all values in a Numpy array have the same data type.  Numpy arrays are indexed by a sequence of zero-based integer positions -- that is, `x[0]` is the first element of the 1-d array `x`. The number of axes (dimensions) is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.
# 
# Below are some one-line expressions that illustrate basic usage of Numpy.

# In[4]:


### Create a rank-1 numpy array with 1 axes of length 3.
a = np.array([1, 2, 3])

### Print object type
print("type(a) =", type(a))

### Print shape
print("\na.shape =", a.shape)

### Print some values in a
print("\nValues in a: ", a[0], a[1], a[2])

### Create a 2x2 numpy array
b = np.array([[1, 2], [3, 4]])

### Print shape
print("\nb.shape =", b.shape)

## Print some values in b
print("\nValues in b: ", b[0, 0], b[0, 1], b[1, 1])

### Create a 3x2 numpy array
c = np.array([[1, 2], [3, 4], [5, 6]])

### Print shape
print("\nc.shape =", c.shape)

### Print some values in c
print("\nValues in c: ", c[0, 1], c[1, 0], c[2, 0], c[2, 1])


# In[5]:


### 2x3 array containing zeros 
d = np.zeros((2, 3))
print("d =\n", d)

### 4x2 array of ones
e = np.ones((4, 2))
print("\ne =\n", e)

### 2x2 constant array
f = np.full((2, 2), 9)
print("\nf =\n", f)

### 3x3 random array
g = np.random.random((3, 3))
print("\ng =\n", g)

### 2x2 array with uninitialized values
h = np.empty((2, 2))
print("\nh =\n", h)


# #### Array Indexing and aliasing
# 
# It is important to note that Python arrays may share memory, in which case changing the values in one array may alter values in another array.

# In[6]:


### Create 3x4 array
h = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print("h=\n", h)

### Slice array to make a 2x2 sub-array
i = h[:2, 1:3]

print("\ni=\n", i)

print("\nh[0, 1] =", h[0, 1])

### Modify the slice
i[0, 0] = 1738

### Print to show how modifying the slice also changes the base object
print("\nh[0, 1] =", h[0, 1])


# If you want to be sure that two arrays do not share memory, use the `copy` method:

# In[7]:


h = np.zeros((3, 3))
i = h[0:2, 0:2].copy()
h[0, 0] = 99
print("h =\n", h)
print("\ni =\n", i)


# #### Datatypes
# 
# Numpy arrays are homogeneous, and we can retrieve the data type shared by all elements using the `dtype` attribute.

# In[8]:


### Integer
j = np.array([1, 2])
print(j.dtype)  

### Float
k = np.array([1.0, 2.0])
print(k.dtype)         

### Force Data Type
l = np.array([1.0, 2.0], dtype=np.int64)
print(l.dtype)


# Right now you don't need to know a lot about the different numeric data types.  Briefly, `int64` refers to a 64-bit or 8-byte signed integer while `float64` refers to a 64 bit "floating point" value which can approximate any real number.

# #### Array arithmetic
# 
# Basic mathematical functions operate element-wise on arrays, and are available both using operator symbols (+, -, etc.) and as functions in the numpy module:

# In[9]:


x = np.array([[1, 2],[3, 4]], dtype=np.float64)
y = np.array([[5, 6],[7, 8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print("x + y =\n", x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print("\nx - y =\n", x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print("\nx * y =\n", x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print("\nx / y =\n", x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print("\nsqrt(x) =\n", np.sqrt(x))


# In[10]:


x = np.array([[1,2],[3,4]])
print("x =\n", x)

### Compute sum of all elements; prints "10"
print("\nsum(x) =", np.sum(x))

### Compute sum of each column; prints "[4 6]"
print("sum(x, axis=0) =", np.sum(x, axis=0)) 

### Compute sum of each row; prints "[3 7]"
print("sum(x, axis=1) =", np.sum(x, axis=1))


# In[11]:


x = np.array([[1,2],[3,4]])
print("x =\n", x)

### Compute mean of all elements; prints "2.5"
print("\nmean(x) =", np.mean(x))

### Compute mean of each column; prints "[2 3]"
print("mean(x, axis=0) =", np.mean(x, axis=0)) 

### Compute mean of each row; prints "[1.5 3.5]"
print("mean(x, axis=1) =", np.mean(x, axis=1))


# # Data management with Pandas
# 
# Numpy is useful for mathematical calculations in which everything is a number.  In data science we often deal with heterogeneous data including numbers, text, and time values.  Pandas is a library that provides functionality for working with the type of data that frequently arises in real-world data science.  Pandas provides functionality for manipulating data (e.g. transforming values and selecting subsets), summarizing data, reading data to and from files, among many other tasks.
# 
# The main data structure that Pandas works with is called a [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).  This is a two-dimensional table of data in which the rows typically represent cases or observations (e.g. Cartwheel Contest Participants), and the columns represent variables.  Pandas also has a one-dimensional data structure called a [Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) that we will encounter when accesing a single column of a Data Frame.
# 
# Pandas has a variety of functions named '`read_xxx`' for reading data in different formats from "static" sources such as files.  Right now we will focus on reading '`csv`' files, where "csv" stands for "comma-separated values". A csv file is a lot like a spreadsheet, but it is stored in text form, using commas to "delimit" the values in a given row.  Other important file formats include excel, json, and sql just to name a few.
# 
# This is a link to the .csv that we will be exploring in this tutorial: [Cartwheel Data](https://www.coursera.org/learn/understanding-visualization-data/resources/0rVxx) (this link leads to the dataset section of the Resources for this course).  You do not need to download this datafile now since it is already available in Coursera's Jupyter environment. 
# 
# There are many other options to '`read_csv`' that are very useful.  For example, you would use the option `sep='\t'` instead of the default `sep=','` if the fields of your data file were delimited by tabs instead of commas.  See [here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) for the full documentation for '`read_csv`'.

# ### Importing data

# In[12]:


# The file name string that holds our .csv file
fname = "Cartwheeldata.csv"

# Read the .csv file and store it as a Pandas Data Frame
df = pd.read_csv(fname)

# Print the object type
type(df)


# ### Viewing data

# We can view the top few rows of our Data Frame by calling the head() method.

# In[13]:


df.head()


# The head() method simply shows the first 5 rows of our Data Frame.  If you want to see, say, the first 10 rows of data, you would pass '10' as an argument to the head method:

# In[14]:


df.head(10)


# As you can see, we have a 2-dimensional table of values, where each row is an observation in our cartwheel data, and each column is a variable describing some characteristic of the participants.
# 
# To see the column names, access the 'columns' attribute of the data frame:

# In[15]:


df.columns


# In a dataframe, each column has a single type, but different columns may have different types.  This is important since datasets in the real world contain variables that may have different types, but within a variable, all observations should have the same type. Access the dtypes attribute of the dataframe to see the data type of each column. 

# In[16]:


df.dtypes


# ### Slicing data frames
# 
# Like any table, the rows and columns of a Pandas Data Frame can be referred to by position.  Since Python always counts from 0, the rows and columns are numbered 0, 1, 2, etc.
# 
# Pandas Data Frames also have row and column "indexes" that may be more natural to use than numeric positions in many cases.  For example, if our Data Frame contains information about people, we may have a column named "Age".  Although we may know that the age column is in position 3 (the fourth column due to the zero-based indexing), it is generally preferable to access this column by its name ("Age") rather than by its position (3).  One reason for this is that we may at some point manipulate the Data Frame so that the column positions change.
# 
# The default index values are simply the positions.  In many cases we do not replace the default row indices with another index, so there is no meaningful difference between label-based and position based row operations.  But most datasets have informative column names, so it is uncommon to encounter a Data Frame that uses the default column indices.
# 
# The most common ways to index and select values from Pandas Data Frames are fairly straightforward, but there are also many advanced indexing techniques. See [here](https://pandas.pydata.org/docs/user_guide/indexing.html) for a more complete treatment of this topic.
# 
# There are three main ways to "slice" a Data Frame.
# 
# 1. .loc() -- select based on index values
# 2. .iloc() -- select based on positions
# 3. .ix()
# 
# Here we will cover the .loc() and .iloc() slicing functions.
# 
# ### Indexing with .loc()
# The .loc() method for a Data Frame takes two indexing values separated by ','. The first indexing value selects rows and the second indexing value selects columns.  An indexing value may be a single index value, a range of index values, or a list containing one or more index values.  Below we provide examples to cover some of the more common use-cases:

# In[17]:


# Return all observations of the variable CWDistance
df.loc[:,"CWDistance"]


# The following syntax is equivalent to what we used above:

# In[18]:


df["CWDistance"]


# The following syntax is also equivalent to the preceeding two examples, but is somewhat dispreferred (in rare cases using this syntax can lead to collisions between method names and variable names, and this syntax does not work if you have variable names that include whitespace or punctuation symbols).

# In[19]:


df.CWDistance


# In the following example we select all rows for multiple columns, ["CWDistance", "Height", "Wingspan"]:

# In[20]:


df.loc[:,["CWDistance", "Height", "Wingspan"]]


# The syntax below is equivalent:

# In[21]:


df[["CWDistance", "Height", "Wingspan"]]


# In the example below we select a limited range of rows for multiple columns, ["CWDistance", "Height", "Wingspan"].  Note that we are using the default row index values which coincide with the row positions.

# In[22]:


df.loc[:9, ["CWDistance", "Height", "Wingspan"]]


# Below we select a limited range of rows for all columns:

# In[23]:


df.loc[10:15]


# The .loc() function requires two arguments, the indices of the rows and the column names you wish to observe.
# 
# In the above case **:** specifies all rows, and our column is **CWDistance**. df.loc[**:**,**"CWDistance"**]

# Now, let's say we only want to return the first 10 observations:

# In[24]:


df.loc[:9, "CWDistance"]


# ### Indexing with .iloc()
# 
# The .iloc() method is used for position-based slicing. Recall that Python uses zero-based indexing, so the first value is in position zero.  Here are some examples:

# In[25]:


df.iloc[:4]


# In[26]:


df.iloc[1:5, 2:4]


# In the next example, we mix position-based slicing in the rows and label-based indexing in the columns:

# In[27]:


df.iloc[1:5, :][["Gender", "GenderGroup"]]


# We can view the data types of our data frame columns with by viewing the .dtypes attribute of our data frame:

# In[28]:


df.dtypes


# The result indicates we have integers, floats, and objects with our Data Frame.  A variable with "object" dtype often contains strings, but may in some cases contain other values that are "wrapped" as Python values.
# 
# We may also want to observe the different unique values within a specific column, let's do this for the `Gender` variable:

# In[34]:


# List unique values in the df['Gender'] column
df["Gender"].unique()


# There is another variable called `GenderGroup`, let's consider this variable as well:

# In[35]:


df["GenderGroup"].unique()


# It seems that these two variables may contain redundant information. Let's explore this further by displaying only these two columns:

# In[31]:


df[["Gender", "GenderGroup"]]


# From inspecting this output, it seems that these variables contain the same information in different coding schemes.  We can further confirm this hunch using the [crosstab](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html) function in Pandas:

# In[32]:


pd.crosstab(df["Gender"], df["GenderGroup"])


# From the result above, it is clear that everyone whose Gender is "F" has a GenderGroup value of 1, and everyone whose gender is "M" has a GenderGroup value of 2.
# 
# The same result can be obtained using the groupby() and size() methods:

# In[33]:


df.groupby(['Gender','GenderGroup']).size()


# Again, the output indicates that we have two combinations:
# 
# * Case 1: Gender = F & Gender Group = 1 
# * Case 2: Gender = M & GenderGroup = 2.  
