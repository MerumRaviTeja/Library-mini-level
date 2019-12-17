#https://www.tutorialspoint.com/numpy/numpy_array_attributes.htm
#https://www.guru99.com/numpy-array.html

import numpy as np

l = [1,2,3]
print(l.append(4))
a = np.array([1,2,3,4])
print(a)

l2 = l + l
print('+ in python list is Concatination :',l2)
print('+ in numpy is 1 D Addition',a+a)

print(a*2) #o/p [2,4,6]
print(l*2) #o/p
#Note : if python list of each element multiplication we need loop each element and multiply
L = []
for e in l:
    L.append(e*e)
print('List element Multiply',L)
print(a*a)

# Note : numpy arthemetic operation will do element wise
# Numpy most functions act element wise
print(np.sqrt(a))
print(np.log(a))
print(np.power(a,a))


#Note : for same above function if you want apply on python list
# you need to loop each element and apply function
#Summary : if you want to represent a vector numpy array is littele
# convinient. numpy array as like vector. for loop in python is very slow you can avoid this by using numpy.

#forloop VS cosin VS dot function

a = np.array([1,2])
b = np.array([2,1])
dot = 0
#loop
for e,f in zip(a,b):
    dot = e*f
    print(dot)
print(dot)

print (a*b)
print(a+b)
#print(np.sum(a,b))
print((a*b).sum())

# dot
print(np.dot(a,b))  #o/p :4
print(a.dot(b)) #o/p :4
print(b.dot(a)) #o/p :4

amp = np.sqrt((a*b).sum())
print(amp)
am = np.linalg.norm(a)
print(am)

# Note :  a.b = aTb = sigm d=1 to n  ad*bd
# magnitude of a time the magnitude of b is : a*b = |a||b|cos(o)ab ==> cos(o)ab=aTb/|a||b|

# Note : dot product is speet
# dot speed test example

from datetime import datetime
a = np.random.randn(100)
b = np.random.randn(100)
T = 100000

def slow_dot_product(a,b):
    result = 0
    for e,f in zip(a,b):
        result +=e*f
    return result

to = datetime.now()
for t in range(T):
    slow_dot_product(a,b)
dt1 = datetime.now()- to

to = datetime.now()
for t in range(T):
    a.dot(b)

dt2 = datetime.now()- to
print('dt1.total_seconds()',dt1.total_seconds())
print('dt2.total_seconds()',dt2.total_seconds())
print('dt1/dt2',dt1.total_seconds()/dt2.total_seconds())

# Vector and Matrics
# Matric is 2-D array or List in List
M =np.array([[1,2],[3,4]])  # arrasy size shoud be same
print(M)
l = [[1,2],[3,4]]  # list in list

print('index 0 :',l[0])
print('index 0,column 0 :',l[0][0])
print('index 0,column 0 :',l[0][1])
print('index 0,column 0 :',l[1][0])
print('index 0,column 0 :',l[1][1])

print('index 0 :',M[0][0])
print('index 0 :',M[0][1])
print('index 0 :',M[1][0])
print('index 0 :',M[1][1])

m1 = np.matrix('1 2; 3 4')
print(m1)

m2= np.matrix([[1, 2], [3, 4]])
print(m2)
print(m1.T)  # transpose

a = np.array(m2)
print(a)

# summary
# Matrix is 2-D numpy array vector is 1-D numpy array
#Matic is lika a 2-D vector
# matric is 2-D mathamatical object that contain numbers
# vector is 1-D mathamatical object that contain numbers

#Generate Matics to work with
z = np.zeros(10)
print(z)

z = np.zeros((3,3)) # 10X10 matics
o = np.ones((3,3)) # 3X3 matics
print(z)
print(o)

r = np.random.random((3,3)) # random will give uniformaly distributed  numbers between 0 to 1
print(r)

g = np.random.randn(3,3)  # these numbers are Gasian distribution mean =0 and variance 1
print(g)

# Matric Operations
#=================
x = np.array([[1,2],[3,4]])
y = np.linalg.inv(x)
print ('x', x)
print ('inverse :',y)
print (np.dot(x,y))

print('y.dot(x) : ', y.dot(x))

print('np.diag(x).sum() :',np.diag(x).sum())
print(np.trace(x))
import numpy as np

l = [1,2,3]
print(l)

l = l.append(5)


# numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
a = np.array([1, 2, 3])
print(a)

print(a.ndim)

# more than one dimensions
a = np.array([[1, 2], [3, 4]])
print('2dim', a)

# minimum dimensions
a = np.array([1, 2, 3, 4, 5], ndmin=2)
print(a)

# dtype parameter
a = np.array([1, 2, 3], dtype=complex)
print(a)

# numpy.dtype(object, align, copy)
# Object − To be converted to data type object
# Align − If true, adds padding to the field to make it similar to C-struct
# Copy − Makes a new copy of dtype object. If false, the result is reference to builtin data type object

dt = np.dtype(np.int32)
print(dt)

# int8, int16, int32, int64 can be replaced by equivalent string 'i1', 'i2','i4', etc.
dt = np.dtype('i4')
print(dt)

dt = np.dtype([('age', np.int8)])
print(dt)

dt = np.dtype([('age', np.int8)])
a = np.array([(10,), (20,), (30,)], dtype=dt)
print(a)

student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
print(a['name'])

#ndarray.shape    # This array attribute returns a tuple consisting of array dimensions. It can also be used to resize the array.
a = np.array([[1,2,3],[4,5,6]])
print (a.shape)

a = np.array([[1,2,3],[4,5,6]])
a.shape = (3,2)
print(a)

a = np.array([[1,2,3],[4,5,6]])
b = a.reshape(3,2) #NumPy also provides a reshape function to resize an array.
print(b)


#ndarray.ndim --This array attribute returns the number of array dimensions.

a = np.arange(24)
print(a)
print(a.ndim)

b = a.reshape(2,4,3)
print(b)

c = a.reshape(3,2,4)
print(c)

# numpy.itemsize : This array attribute returns the length of each element of array in bytes.
x = np.array([1,2,3,4,5], dtype = np.int8)
print (x.itemsize)

x = np.array([1,2,3,4,5], dtype = np.float32)
print(x.itemsize)


#numpy.flags -----The ndarray object has the following attributes. Its current values are returned by this function.

#1  C_CONTIGUOUS (C) The data is in a single, C-style contiguous segment

#2	F_CONTIGUOUS (F) The data is in a single, Fortran-style contiguous segment

#3	OWNDATA (O) The array owns the memory it uses or borrows it from another object

#4  WRITEABLE (W) The data area can be written to. Setting this to False locks the data, making it read-only

#5	ALIGNED (A) The data and all elements are aligned appropriately for the hardware

#6	 UPDATEIFCOPY (U) ---This array is a copy of some other array. When this array is deallocated, the base array will be updated with the contents of this array

x = np.array([1,2,3,4,5])
print(x.flags)


#numpy.empty ---It creates an uninitialized array of specified shape and dtype. It uses the following constructor −

#numpy.empty(shape, dtype = float, order = 'C')
#1  Shape  Shape of an empty array in int or tuple of int
#2 Dtype ---Desired output data type. Optional
#3	Order 'C' for C-style row-major array, 'F' for FORTRAN style column-major array

x = np.empty([3,2], dtype = int)
print(x)

#numpy.zeros  --Returns a new array of specified size, filled with zeros.
#numpy.zeros(shape, dtype = float, order = 'C')

x = np.zeros(8)
print(x)
print(x.reshape(2,2,2))

x = np.zeros((5,), dtype = np.int)
print(x)

x = np.zeros((2,2,2), dtype = np.int)
print(x)

# custom type
x = np.zeros((2,2), dtype = [('x', 'i4'), ('y', 'i4')])
print (x)

#numpy.ones  Returns a new array of specified size and type, filled with ones.
# numpy.ones(shape, dtype = None, order = 'C')

#1 Shape ---Shape of an empty array in int or tuple of int

#2	Dtype --Desired output data type. Optional

#3	Order --'C' for C-style row-major array, 'F' for FORTRAN style column-major array

x = np.ones(5)
print(x)
x = np.ones([2,2], dtype = int)
print (x)

#numpy.asarray
#This function is similar to numpy.array except for the fact that it has fewer parameters. This routine is useful for
# converting Python sequence into ndarray.

#numpy.asarray(a, dtype = None, order = None)

#1 a  --Input data in any form such as list, list of tuples, tuples, tuple of tuples or tuple of lists

#2	---dtype ---By default, the data type of input data is applied to the resultant ndarray

#3 order---C (row major) or F (column major). C is default

# convert list to ndarray
x = [1,2,3]
a = np.asarray(x)
print (a)
print (a*3)
print(x*2)


# dtype is set
x = [1,2,3]
a = np.asarray(x, dtype = float)
print(a)

# ndarray from tuple
x = (1,2,3)
a = np.asarray(x)
print (a)

# ndarray from list of tuples
x = [(1,2,3),(4,5)]
a = np.asarray(x)
print (a)

#numpy.frombuffer
# This function interprets a buffer as one-dimensional array. Any object that exposes the buffer interface is used as parameter to return an ndarray.

#numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)

#1	buffer ---Any object that exposes buffer interface
#2	dtype---Data type of returned ndarray. Defaults to float
#3	count---The number of items to read, default -1 means all data
#4	offset---The starting position to read from. Default is 0
"""
s = 'Hello World'
a = np.frombuffer(s, dtype='S1')
print(a)
"""
#numpy.fromiter  This function builds an ndarray object from any iterable object. A new one-dimensional array is returned by this function.
#numpy.fromiter(iterable, dtype, count = -1)

#1	iterable--Any iterable object
#2	dtype --Data type of resultant array
#3	count---The number of items to be read from iterator. Default is -1 which means all data to be read

list = range(5)
print (list)

# obtain iterator object from list
list = range(5)
it = iter(list)

# use iterator to create ndarray
x = np.fromiter(it, dtype = float)
print (x)

#numpy.arange
#This function returns an ndarray object containing evenly spaced values within a given range. The format of the function is as follows −

#numpy.arange(start, stop, step, dtype)

x = np.arange(5)
print(x)

# dtype set
x = np.arange(5, dtype = float)
print (x)

# start and stop parameters set
x = np.arange(2,20,2)
print (x)

#numpy.linspace
#This function is similar to arange() function. In this function, instead of step size, the number of evenly spaced values between the interval is specified. The usage of this function is as follows −
#numpy.linspace(start, stop, num, endpoint, retstep, dtype)

#1	start--The starting value of the sequence
#2   stop --The end value of the sequence, included in the sequence if endpoint set to true
#3	num---The number of evenly spaced samples to be generated. Default is 50
#4   endpoint----True by default, hence the stop value is included in the sequence. If false, it is not included
#5	retstep---If true, returns samples and step between the consecutive numbers
#6   dtype---Data type of output ndarray

x = np.linspace(10,20,5)
print(x)

# endpoint set to false
x = np.linspace(10,20, 5, endpoint = False)
print (x)

# find retstep value
x = np.linspace(1,2,5, retstep = True)
print (x)
# retstep here is 0.25

#numpy.logspace
#This function returns an ndarray object that contains the numbers that are evenly spaced on a log scale. Start and stop endpoints of the scale are indices of the base, usually 10.
#1	start--The starting point of the sequence is basestart
#2	stop--The final value of sequence is basestop
#3	num--The number of values between the range. Default is 50
#4	endpoint---If true, stop is the last value in the range
#5	base--Base of log space, default is 10
#6	dtype--Data type of output array. If not given, it depends upon other input arguments


# default base is 10
a = np.logspace(1.0, 2.0, num = 10)
print(a)

# set base of log space to 2
a = np.logspace(1,10,num = 10, base = 2)
print (a)

#Indexing & Slicing
#===================
#Contents of ndarray object can be accessed and modified by indexing or slicing, just like Python's in-built container objects.
#As mentioned earlier, items in ndarray object follows zero-based index. Three types of indexing methods are available − field access, basic slicing and advanced indexing.
#Basic slicing is an extension of Python's basic concept of slicing to n dimensions. A Python slice object is constructed by giving start, stop, and step parameters to the built-in slice function. This slice object is passed to the array to extract a part of array.

a = np.arange(10)
s = slice(2,7,2)
print (a[s])

#oputu : [2  4  6]
#In the above example, an ndarray object is prepared by arange() function. Then a slice object is defined with start, stop, and step values 2, 7, and 2 respectively. When this slice object is passed to the ndarray, a part of it starting with index 2 up to 7 with a step of 2 is sliced.
#The same result can also be obtained by giving the slicing parameters separated by a colon : (start:stop:step) directly to the ndarray object.

a = np.arange(10)
b = a[2:7:2]
print(b)


# slice single item
a = np.arange(10)
b = a[5]
print(b)

# slice items starting from index
import numpy as np
a = np.arange(10)
print(a[2:])

# slice items between indexes
import numpy as np
a = np.arange(10)
print(a[2:5])

#same we can applies to multi-dimensional ndarray too.
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)

# slice items starting from index
print ('Now we will slice the array from the index a[1:]')
print(a[1:])

#Slicing can also include ellipsis (…) to make a selection tuple of the same length as the dimension of an array. If ellipsis is used at the row position, it will return an ndarray comprising of items in rows.
import numpy as np
a = np.array([[1,2,3],[3,4,5],[4,5,6]])

print('Our array is:')
print (a)
print('\n')

# this returns array of items in the second column
print ('The items in the second column are:')
print (a[...,1])
print('\n')

# Now we will slice all items from the second row
print('The items in the second row are:')
print (a[1,...])
print ('\n')

# Now we will slice all items from column 1 onwards
print('The items column 1 onwards are:')
print(a[...,1:])

#There are two types of advanced indexing − Integer and Boolean.
#This mechanism helps in selecting any arbitrary item in an array based on its Ndimensional index. Each integer array represents the number of indexes into that dimension. When the index consists of as many integer arrays as the dimensions of the target ndarray, it becomes straightforward.
#In the following example, one element of specified column from each row of ndarray object is selected. Hence, the row index contains all row numbers, and the column index specifies the element to be selected.
x = np.array([[1, 2], [3, 4], [5, 6]])
y = x[[0,1,2], [0,1,0]]
print(y)

#The selection includes elements at (0,0), (1,1) and (2,0) from the first array.
#In the following example, elements placed at corners of a 4X3 array are selected. The row indices of selection are [0, 0] and [3,3] whereas the column indices are [0,2] and [0,2].

#example2
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])

print('Our array is:')
print(x)
print('\n')

rows = np.array([[0, 0], [3, 3]])
print('rows',rows)
cols = np.array([[0, 2], [0, 2]])
print('col',cols)
y = x[rows, cols]

print('The corner elements of this array are:')
print(y)

#example 3

x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])

print('Our array is:')
print (x)
print ('\n')

# slicing
z = x[1:4,1:3]

print ('After slicing, our array becomes:')
print (z)
print ('\n')

# using advanced index for column
y = x[1:4,[1,2]]

print ('Slicing using advanced index for column:')
print(y)

#exaples
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])

print('Our array is:')
print(x)
print('\n')

# Now we will print the items greater than 5
print ('The items greater than 5 are:')
print (x[x > 5])

a = np.array([np.nan, 1,2,np.nan,3,4,5])
print(a[~np.isnan(a)])

a = np.array([1, 2+6j, 5, 3.5+5j])
print(a[np.iscomplex(a)])

#Broadcasting
#=============
#The term broadcasting refers to the ability of NumPy to treat arrays of different shapes during arithmetic operations. Arithmetic operations on arrays are usually done on corresponding elements. If two arrays are of exactly the same shape, then these operations are smoothly performed.

a = np.array([1,2,3,4])
b = np.array([10,20,30,40])
c = a * b
print(c)
#example2
#If the dimensions of two arrays are dissimilar, element-to-element operations are not possible. However, operations on arrays of non-similar shapes is still possible in NumPy, because of the broadcasting capability. The smaller array is broadcast to the size of the larger array so that they have compatible shapes.
import numpy as np
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]])
b = np.array([1.0,2.0,3.0])

print('First array:')
print(a)
print('\n')

print('Second array:')
print(b)
print('\n')

print('First Array + Second Array')
print(a + b)

# Iterating Over Array
#=====================
#NumPy package contains an iterator object numpy.nditer. It is an efficient multidimensional iterator object using which it is possible to iterate over an array. Each element of an array is visited using Python’s standard Iterator interface.

#Let us create a 3X4 array using arange() function and iterate over it using nditer.

a = np.arange(0,60,5)
a = a.reshape(3,4)

print('Original array is:')
print(a)
print('\n')

print('Modified array is:')
for x in np.nditer(a):
   print(x,)

#Example 2 :The order of iteration is chosen to match the memory layout of an array, without considering a particular ordering. This can be seen by iterating over the transpose of the above array.

a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print(c)

print('Original array is:')
print(a)
print('\n')

print('Transpose of the original array is:')
b = a.T
print(b)
print('\n')

print('Modified array is:')
for x in np.nditer(b):
    print(x,)

#Iteration Order : If the same elements are stored using F-style order, the iterator chooses the more efficient way of iterating over an array.
a = np.arange(0,60,5)
a = a.reshape(3,4)
print ('Original array is:')
print(a)
print('\n')

print('Transpose of the original array is:')
b = a.T
print (b)
print ('\n')

print ('Sorted in C-style order:')
c = b.copy(order = 'C')
print (c)
for x in np.nditer(c):
   print (x,)

print ('\n')

print ('Sorted in F-style order:')
c = b.copy(order = 'F')
print (c)
for x in np.nditer(c):
   print(x,)

#It is possible to force nditer object to use a specific order by explicitly mentioning it.

a = np.arange(0,60,5)
a = a.reshape(3,4)

print('Original array is:')
print(a)
print('\n')

print('Sorted in C-style order:')
for x in np.nditer(a, order = 'C'):
   print(x,)
print ('\n')

print('Sorted in F-style order:')
for x in np.nditer(a, order = 'F'):
   print(x,)

#Modifying Array Values
a = np.arange(0,60,5)
a = a.reshape(3,4)
print('Original array is:')
print (a)
print('\n')

for x in np.nditer(a, op_flags = ['readwrite']):
   x[...] = 2*x
print('Modified array is:')
print(a)

#External Loop : The nditer class constructor has a ‘flags’ parameter, which can take the following values −
#1	c_index---C_order index can be tracked
#2	f_index--Fortran_order index is tracked
#3	multi-index --Type of indexes with one per iteration can be tracked
#4	external_loop--Causes values given to be one-dimensional arrays with multiple values instead of zero-dimensional array

#Example--In the following example, one-dimensional arrays corresponding to each column is traversed by the iterator.
a = np.arange(0,60,5)
a = a.reshape(3,4)

print ('Original array is:')
print (a)
print ('\n')

print('Modified array is:')
for x in np.nditer(a, flags = ['external_loop'], order = 'F'):
   print(x,)

#Broadcasting Iteration : If two arrays are broadcastable, a combined nditer object is able to iterate upon them concurrently. Assuming that an array a has dimension 3X4, and there is another array b of dimension 1X4, the iterator of following type is used (array b is broadcast to size of a).
#Example
a = np.arange(0,60,5)
a = a.reshape(3,4)

print ('First array is:')
print (a)
print ('\n')

print ('Second array is:')
b = np.array([1, 2, 3, 4], dtype = int)
print(b)
print('\n')

print('Modified array is:')
for x,y in np.nditer([a,b]):
   print ("%d:%d" % (x,y),)

#Array Manipulation
#====================

#Changing Shape :
#1	reshape :Gives a new shape to an array without changing its data
#2	flat :A 1-D iterator over the array
#3	flatten : Returns a copy of the array collapsed into one dimension
#4	ravel :Returns a contiguous flattened array

#Transpose Operations :
#1	transpose :Permutes the dimensions of an array
#2	ndarray.T :Same as self.transpose()
#3	rollaxis :Rolls the specified axis backwards
#4	swapaxes :Interchanges the two axes of an array

#Changing Dimensions:
#1	broadcast :Produces an object that mimics broadcasting
#2	broadcast_to :Broadcasts an array to a new shape
#3	expand_dims : Expands the shape of an array
#4	squeeze :Removes single-dimensional entries from the shape of an array

#Joining Arrays:
#1	concatenate :Joins a sequence of arrays along an existing axis
#2	stack :Joins a sequence of arrays along a new axis
#3	hstack : Stacks arrays in sequence horizontally (column wise)
#4	vstack :Stacks arrays in sequence vertically (row wise)

#Splitting Arrays:
#1	split :Splits an array into multiple sub-arrays
#2	hsplit :Splits an array into multiple sub-arrays horizontally (column-wise)
#3	vsplit :Splits an array into multiple sub-arrays vertically (row-wise)

#Adding / Removing Elements
#1	resize :Returns a new array with the specified shape
#2	append :Appends the values to the end of an array
#3	insert :Inserts the values along the given axis before the given indices
#4	delete :Returns a new array with sub-arrays along an axis deleted
#5	unique :Finds the unique elements of an array

#Binary Operators
#=================
#1	bitwise_and :Computes bitwise AND operation of array elements
#2	bitwise_or :Computes bitwise OR operation of array elements
#3	invert :Computes bitwise NOT
#4	left_shift :Shifts bits of a binary representation to the left
#5	right_shift :Shifts bits of binary representation to the right

#Mathematical Functions
#=========================

a = np.array([0,30,45,60,90])

print ('Sine of different angles:')
# Convert to radians by multiplying with pi/180
print(np.sin(a*np.pi/180) )
print ('\n')

print('Cosine values for angles in array:')
print(np.cos(a*np.pi/180))
print('\n')

print('Tangent values for given angles:')
print(np.tan(a*np.pi/180))

#rounding values
a = np.array([1.0,5.55, 123, 0.567, 25.532])

print('Original array:')
print(a)
print('\n')

print('After rounding:')
print(np.around(a))
print(np.around(a, decimals = 1))
print(np.around(a, decimals = -1))

#Arithmetic Operations
#======================
#nput arrays for performing arithmetic operations such as add(), subtract(), multiply(), and divide() must be either of the same shape or should conform to array broadcasting rules.
a = np.arange(9, dtype = np.float_).reshape(3,3)

print('First array:')
print(a)
print('\n')

print('Second array:')
b = np.array([10,10,10])
print(b)
print('\n')

print('Add the two arrays:')
print(np.add(a,b))
print('\n')

print('Subtract the two arrays:')
print(np.subtract(a,b))
print('\n')

print('Multiply the two arrays:')
print(np.multiply(a,b))
print('\n')

print('Divide the two arrays:')
print(np.divide(a,b))

#np.power
a = np.array([10,100,1000])

print('Our array is:')
print(a)
print('\n')

print('Applying power function:')
print(np.power(a,2))
print('\n')

print('Second array:')
b = np.array([1,2,3])
print(b)
print('\n')

print('Applying power function again:')
print(np.power(a,b))

# numpy.mod() : This function returns the remainder of division of the corresponding elements in the input array. The function numpy.remainder() also produces the same result.
a = np.array([10,20,30])
b = np.array([3,5,7])

print('First array:')
print(a)
print('\n')

print('Second array:')
print(b)
print('\n')

print('Applying mod() function:')
print(np.mod(a,b))
print('\n')

print('Applying remainder() function:')
print(np.remainder(a,b))

#Statistical Functions
#=====================
#These functions return the minimum and the maximum from the elements in the given array along the specified axis.

a = np.array([[3,7,5],[8,4,3],[2,4,9]])

print('Our array is:')
print(a)
print('\n')

print('Applying amin() function:')
print(np.amin(a,1))
print('\n')

print('Applying amin() function again:')
print(np.amin(a,0))
print ('\n')

print('Applying amax() function:')
print(np.amax(a))
print('\n')

print('Applying amax() function again:')
print(np.amax(a, axis = 0))

#numpy.ptp() :The numpy.ptp() function returns the range (maximum-minimum) of values along an axis.
a = np.array([[3,7,5],[8,4,3],[2,4,9]])

print ('Our array is:')
print (a)
print ('\n')

print ('Applying ptp() function:')
print (np.ptp(a) )
print ('\n')

print ('Applying ptp() function along axis 1:')
print (np.ptp(a, axis = 1) )
print ('\n')

print('Applying ptp() function along axis 0:')
print (np.ptp(a, axis = 0) )

#numpy.percentile() : Percentile (or a centile) is a measure used in statistics indicating the value below which a given percentage of observations in a group of observations fall. The function numpy.percentile() takes the following arguments
#numpy.percentile(a, q, axis)
#1	a :Input array
#2	q : The percentile to compute must be between 0-100
#3	axis :The axis along which the percentile is to be calculated
a = np.array([[30,40,70],[80,20,10],[50,90,60]])
print ('Our array is:')
print (a )
print ('\n')

print ('Applying percentile() function:')
print(np.percentile(a,50) )
print ('\n')

print('Applying percentile() function along axis 1:')
print(np.percentile(a,50, axis = 1) )
print('\n')

print('Applying percentile() function along axis 0:')
print(np.percentile(a,50, axis = 0))

#numpy.median() : Median is defined as the value separating the higher half of a data sample from the lower half. The numpy.median() function is used as shown in the following program.
a = np.array([[30, 65, 70], [80, 95, 10], [50, 90, 60]])

print('Our array is:')
print(a)
print('\n')

print('Applying median() function:')
print(np.median(a))
print('\n')

print('Applying median() function along axis 0:')
print(np.median(a, axis=0))
print('\n')

print('Applying median() function along axis 1:')
print(np.median(a, axis=1))

#numpy.mean() : Arithmetic mean is the sum of elements along an axis divided by the number of elements. The numpy.mean() function returns the arithmetic mean of elements in the array. If the axis is mentioned, it is calculated along it.

a = np.array([[1,2,3],[3,4,5],[4,5,6]])

print ('Our array is:' )
print (a)
print ('\n')

print ('Applying mean() function:' )
print (np.mean(a) )
print ('\n')

print ('Applying mean() function along axis 0:')
print (np.mean(a, axis = 0) )
print ('\n')

print ('Applying mean() function along axis 1:')
print (np.mean(a, axis = 1))

#numpy.average()

a = np.array([1,2,3,4])

print ('Our array is:')
print (a )
print ('\n'  )

print ('Applying average() function:' )
print (np.average(a) )
print ('\n')

# this is same as mean when weight is not specified
wts = (np.array([4,3,2,1]) )

print ('Applying average() function again:' )
print (np.average(a,weights = wts) )
print ('\n')

# Returns the sum of weights, if the returned parameter is set to True.
print ('Sum of weights')
print (np.average([1,2,3, 4],weights = [4,3,2,1], returned = True))

#Note : In a multi-dimensional array, the axis for computation can be specified.
a = np.arange(6).reshape(3,2)

print ('Our array is:')
print (a )
print ('\n')

print ('Modified array:' )
wt = np.array([3,5])
print (np.average(a, axis = 1, weights = wt) )
print ('\n')

print ('Modified array:')
print (np.average(a, axis = 1, weights = wt, returned = True))

#Standard Deviation :Standard deviation is the square root of the average of squared deviations from mean. The formula for standard deviation is as follows −
#std = sqrt(mean(abs(x - x.mean())**2))
print (np.std([1,2,3,4]))

print(np.var([1,2,3,4]))

#Sort, Search & Counting Functions
#=================================
#numpy.sort() : The sort() function returns a sorted copy of the input array. It has the following parameters −
#numpy.sort(a, axis, kind, order)

#1	a :Array to be sorted
#2	axis : The axis along which the array is to be sorted. If none, the array is flattened, sorting on the last axis
#3	kind : Default is quicksort
#4	order : If the array contains fields, the order of fields to be sorted

a = np.array([[3, 7], [9, 1]])

print('Our array is:')
print(a)
print('\n')

print('Applying sort() function:')
print(np.sort(a))
print('\n')

print('Sort along axis 0:')
print(np.sort(a, axis=0))
print('\n')

# Order parameter in sort function
dt = np.dtype([('name', 'S10'), ('age', int)])
a = np.array([("raju", 21), ("anil", 25), ("ravi", 17), ("amar", 27)], dtype=dt)

print('Our array is:')
print(a)
print('\n')

print('Order by name:')
print(np.sort(a, order='name'))

#numpy.argsort() :The numpy.argsort() function performs an indirect sort on input array, along the given axis and using a specified kind of sort to return the array of indices of data. This indices array is used to construct the sorted array.

x = np.array([3, 1, 2])

print ('Our array is:')
print (x )
print ('\n')

print ('Applying argsort() to x:')
y = np.argsort(x)
print (y )
print ('\n'  )

print ('Reconstruct original array in sorted order:')
print (x[y] )
print ('\n'  )

print ('Reconstruct the original array using loop:')
for i in y:
   print (x[i],)

#numpy.lexsort() :function performs an indirect sort using a sequence of keys. The keys can be seen as a column in a spreadsheet. The function returns an array of indices, using which the sorted data can be obtained. Note, that the last key happens to be the primary key of sort.

nm = ('raju','anil','ravi','amar')
dv = ('f.y.', 's.y.', 's.y.', 'f.y.')
ind = np.lexsort((dv,nm))

print ('Applying lexsort() function:')
print (ind)
print('\n')

print('Use this index to get sorted data:')
print([nm[i] + ", " + dv[i] for i in ind])

#numpy.argmax() and numpy.argmin() : These two functions return the indices of maximum and minimum elements respectively along the given axis.
a = np.array([[30, 40, 70], [80, 20, 10], [50, 90, 60]])

print('Our array is:')
print(a)
print('\n')

print('Applying argmax() function:')
print(np.argmax(a))
print('\n')

print('Index of maximum number in flattened array')
print(a.flatten())
print('\n')

print('Array containing indices of maximum along axis 0:')
maxindex = np.argmax(a, axis=0)
print(maxindex)
print('\n')

print('Array containing indices of maximum along axis 1:')
maxindex = np.argmax(a, axis=1)
print(maxindex)
print('\n')

print('Applying argmin() function:')
minindex = np.argmin(a)
print(minindex)
print('\n')

print('Flattened array:')
print(a.flatten()[minindex])
print('\n')

print('Flattened array along axis 0:')
minindex = np.argmin(a, axis=0)
print(minindex)
print('\n')

print('Flattened array along axis 1:')
minindex = np.argmin(a, axis=1)
print(minindex)

#numpy.nonzero() : The numpy.nonzero() function returns the indices of non-zero elements in the input array.
a = np.array([[30,40,0],[0,20,10],[50,0,60]])

print('Our array is:')
print(a)
print('\n')

print('Applying nonzero() function:')
print(np.nonzero (a))

#numpy.where() : The where() function returns the indices of elements in an input array where the given condition is satisfied.

x = np.arange(9.).reshape(3, 3)

print('Our array is:')
print (x)

print ('Indices of elements > 3')
y = np.where(x > 3)
print(y)

print ('Use these indices to get elements satisfying the condition')
print(x[y])

#numpy.extract() :The extract() function returns the elements satisfying any condition.

x = np.arange(9.).reshape(3, 3)

print ('Our array is:')
print (x)

# define a condition
condition = np.mod(x,2) == 0

print ('Element-wise value of condition')
print (condition)

print('Extract elements using condition')
print(np.extract(condition, x))

#Byte Swapping
#=============
#We have seen that the data stored in the memory of a computer depends on which architecture the CPU uses. It may be little-endian (least significant is stored in the smallest address) or big-endian (most significant byte in the smallest address).

#numpy.ndarray.byteswap() : The numpy.ndarray.byteswap() function toggles between the two representations: bigendian and little-endian.

a = np.array([1, 256, 8755], dtype = np.int16)

print ('Our array is:')
print (a)

print ('Representation of data in memory in hexadecimal form:')
print (map(hex,a))
# byteswap() function swaps in place by passing True parameter

print ('Applying byteswap() function:')
print (a.byteswap(True))

print ('In hexadecimal form:')
print(map(hex,a))
# We can see the bytes being swapped

#Copies & Views
#===============
#While executing the functions, some of them return a copy of the input array, while some return the view. When the contents are physically stored in another location, it is called Copy. If on the other hand, a different view of the same memory content is provided, we call it as View.

#No Copy
#Simple assignments do not make the copy of array object. Instead, it uses the same id() of the original array to access it. The id() returns a universal identifier of Python object, similar to the pointer in C.

#Furthermore, any changes in either gets reflected in the other. For example, the changing shape of one will change the shape of the other too.

a = np.arange(6)

print ('Our array is:')
print (a)

print ('Applying id() function:')
print (id(a)  )

print ('a is assigned to b:')
b = a
print (b  )

print ('b has same id():')
print (id(b)  )

print ('Change shape of b:')
b.shape = 3,2
print (b)

print ('Shape of a also gets changed:')
print (a)

#View or Shallow Copy : NumPy has ndarray.view() method which is a new array object that looks at the same data of the original array. Unlike the earlier case, change in dimensions of the new array doesn’t change dimensions of the original.

# To begin with, a is 3X2 array
a = np.arange(6).reshape(3,2)

print ('Array a:' )
print (a)

print ('Create view of a:')
b = a.view()
print (b)

print ('id() for both the arrays are different:' )
print ('id() of a:')
print (id(a))
print ('id() of b:')
print (id(b) )

# Change the shape of b. It does not change the shape of a
b.shape = 2,3

print ('Shape of b:' )
print (b )

print ('Shape of a:')
print (a)

#Slice of an array creates a view.

a = np.array([[10,10], [2,3], [4,5]])

print ('Our array is:')
print (a)

print ('Create a slice:')
s = a[:, :2]
print (s)

#Deep Copy :The ndarray.copy() function creates a deep copy. It is a complete copy of the array and its data, and doesn’t share with the original array.

a = np.array([[10,10], [2,3], [4,5]])

print ('Array a is:' )
print (a )

print ('Create a deep copy of a:')
b = a.copy()
print ('Array b is:' )
print (b )

#b does not share any memory of a
print ('Can we write b is a' )
print (b is a )

print ('Change the contents of b:')
b[0,0] = 100

print ('Modified array b:')
print (b)

print ('a remains unchanged:' )
print (a)

#Matrix Library
#==============
#NumPy package contains a Matrix library numpy.matlib. This module has functions that return matrices instead of ndarray objects.

#matlib.empty() :The matlib.empty() function returns a new matrix without initializing the entries. The function takes the following parameters.

#numpy.matlib.empty(shape, dtype, order)

#1	shape : int or tuple of int defining the shape of the new matrix
#2	Dtype :  Optional. Data type of the output
#3	order : C or F

import numpy.matlib
import numpy as np

print (np.matlib.empty((2,2)))
# filled with random data

#numpy.matlib.zeros() : This function returns the matrix filled with zeros.
print (np.matlib.zeros((2,2)))

#numpy.matlib.ones() : This function returns the matrix filled with 1s.

print (np.matlib.ones((2,2)))

#numpy.matlib.eye() : This function returns a matrix with 1 along the diagonal elements and the zeros elsewhere. The function takes the following parameters.
#numpy.matlib.eye(n, M,k, dtype)

#1	n :The number of rows in the resulting matrix
#2	M : The number of columns, defaults to n
#3	k : Index of diagonal
#4	dtype : Data type of the output

print (np.matlib.eye(n = 3, M = 4, k = 0, dtype = float))

#numpy.matlib.identity() : The numpy.matlib.identity() function returns the Identity matrix of the given size. An identity matrix is a square matrix with all diagonal elements as 1.

print (np.matlib.identity(5, dtype = float))

#numpy.matlib.rand() : The numpy.matlib.rand() function returns a matrix of the given size filled with random values.
print (np.matlib.rand(3,3))

#Note that a matrix is always two-dimensional, whereas ndarray is an n-dimensional array. Both the objects are inter-convertible.

i = np.matrix('1,2;3,4')
print (i)

j = np.asarray(i)
print (j)

k = np.asmatrix (j)
print (k)

#Linear Algebra
#===============
#numpy.dot() :This function returns the dot product of two arrays. For 2-D vectors, it is the equivalent to matrix multiplication. For 1-D arrays, it is the inner product of the vectors. For N-dimensional arrays, it is a sum product over the last axis of a and the second-last axis of b.
a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
np.dot(a,b)

#Note that the dot product is calculated as − [[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]

#numpy.vdot() : This function returns the dot product of the two vectors. If the first argument is complex, then its conjugate is used for calculation. If the argument id is multi-dimensional array, it is flattened.

a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
print(np.vdot(a,b))

#Note − 1*11 + 2*12 + 3*13 + 4*14 = 130

#numpy.inner() : This function returns the inner product of vectors for 1-D arrays. For higher dimensions, it returns the sum product over the last axes.
print (np.inner(np.array([1,2,3]),np.array([0,1,0])))
# Equates to 1*0+2*1+3*0

# Multi-dimensional array example
import numpy as np
a = np.array([[1,2], [3,4]])

print ('Array a:')
print(a)
b = np.array([[11, 12], [13, 14]])

print ('Array b:')
print (b)

print ('Inner product:')
print (np.inner(a,b))

#In the above case, the inner product is calculated as −
#   1*11+2*12, 1*13+2*14
#   3*11+4*12, 3*13+4*14

#numpy.matmul() : The numpy.matmul() function returns the matrix product of two arrays. While it returns a normal product for 2-D arrays, if dimensions of either argument is >2, it is treated as a stack of matrices residing in the last two indexes and is broadcast accordingly.
#On the other hand, if either argument is 1-D array, it is promoted to a matrix by appending a 1 to its dimension, which is removed after multiplication.
a = [[1,0],[0,1]]
b = [[4,1],[2,2]]
print(np.matmul(a,b))

# 2-D mixed with 1-D

a = [[1,0],[0,1]]
b = [1,2]
print (np.matmul(a,b))
print (np.matmul(b,a))

# one array having dimensions > 2

a = np.arange(8).reshape(2,2,2)
b = np.arange(4).reshape(2,2)
print (np.matmul(a,b))

#Determinant : Determinant is a very useful value in linear algebra. It calculated from the diagonal elements of a square matrix. For a 2x2 matrix, it is simply the subtraction of the product of the top left and bottom right element from the product of other two.
#In other words, for a matrix [[a,b], [c,d]], the determinant is computed as ‘ad-bc’. The larger square matrices are considered to be a combination of 2x2 matrices.
#The numpy.linalg.det() function calculates the determinant of the input matrix.

a = np.array([[1,2], [3,4]])
print (np.linalg.det(a))

b = np.array([[6,1,1], [4, -2, 5], [2,8,7]])
print (b)
print (np.linalg.det(b))
print (6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - -2*2))

#numpy.linalg.solve() : The numpy.linalg.solve() function gives the solution of linear equations in the matrix form.

#Considering the following linear equations −
#   x + y + z = 6
#    2y + 5z = -4
#     2x + 5y - z = 27

#They can be represented in the matrix form as −

#  $$\begin{bmatrix}1 & 1 & 1 \\0 & 2 & 5 \\2 & 5 & -1\end{bmatrix} \begin{bmatrix}x \\y \\z \end{bmatrix} = \begin{bmatrix}6 \\-4 \\27 \end{bmatrix}$$
#  If these three matrices are called A, X and B, the equation becomes −

#  A*X = B
#    Or
#  X = A-1B

#numpy.linalg.inv() : We use numpy.linalg.inv() function to calculate the inverse of a matrix. The inverse of a matrix is such that if it is multiplied by the original matrix, it results in identity matrix.

x = np.array([[1,2],[3,4]])
y = np.linalg.inv(x)
print (x)
print (y)
print (np.dot(x,y))

#Let us now create an inverse of matrix A in our example.
import numpy as np
a = np.array([[1,1,1],[0,2,5],[2,5,-1]])

print('Array a:')
print(a)
ainv = np.linalg.inv(a)

print ('Inverse of a:')
print (ainv)

print ('Matrix B is:')
b = np.array([[6],[-4],[27]])
print (b)

print ('Compute A-1B:')
x = np.linalg.solve(a,b)
print (x)
# this is the solution to linear equations x = 5, y = 3, z = -2

#I/O with NumPy
a = np.array([1,2,3,4,5])
np.save('outfile',a)
#savetxt()
a = np.array([1,2,3,4,5])
np.savetxt('out.txt',a)
b = np.loadtxt('out.txt')
print (b)

#Matplotlib
#===========
from matplotlib import pyplot as plt

x = np.arange(1,11)
y = 2 * x + 5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y)
#plt.show()
plt.savefig('foo.pdf')
# example3
x = np.arange(1,11)
y = 2 * x + 5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y,"ob")
plt.show()

#Sine Wave Plot : The following script produces the sine wave plot using matplotlib.

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)
plt.title("sine wave form")

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()

# subplot() : The subplot() function allows you to plot different things in the same figure. In the following script, sine and cosine values are plotted.

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

#bar() : The pyplot submodule provides bar() function to generate bar graphs. The following example produces the bar graph of two sets of x and y arrays.

x = [5,8,10]
y = [12,16,6]

x2 = [6,9,11]
y2 = [6,15,7]
plt.bar(x, y, align = 'center')
plt.bar(x2, y2, color = 'g', align = 'center')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()

#Histogram Using Matplotlib
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
np.histogram(a,bins = [0,20,40,60,80,100])
hist,bins = np.histogram(a,bins = [0,20,40,60,80,100])
print (hist)
print (bins)

a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
plt.hist(a, bins = [0,20,40,60,80,100])
plt.title("histogram")
plt.show()

