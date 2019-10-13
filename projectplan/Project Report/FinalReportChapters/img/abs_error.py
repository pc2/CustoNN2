import os
import numpy as np
import json
import sys
from decimal import *

getcontext().prec = 5


# Place your tensorflow path here.
tf_path = '//path/to/tf/output/'
#Place tensorflow filename here.
tf_filename = 'tf_filename.txt'

# place your kernel generated output file name
kernel_path = '/path/to/kernel/output/from/plugin'
# Filename to compare output with
kernel_filename = 'kernel_output_filename.txt'


file1 = open(tf_path+tf_filename,"r")
contents =file1.read()

tf_output = contents.split('\n')

file2 = open(kernel_path+kernel_filename,"r")
kernel_content =file2.read()

kernel_output = kernel_content.split('\n')

##lenght of tf oputpur
print(len(tf_output))
## length of kernel outp-tu
print(len(kernel_output))

abs_error = []


# //(formula measured - actual)/actual

for x in range(0,len(tf_output)-1):
    if(tf_output[x]=='0' or tf_output[x]=='0.0'):
        abs_error.append(abs(Decimal(kernel_output[x])))
    elif(len(tf_output[x])>0):

        try:
            abs_error.append( (abs(Decimal( kernel_output[x] ) ) - abs( Decimal(tf_output[x]) ) ) / abs( Decimal(tf_output[x]) ) )
        except :
            print(x,"error occured")
print(len(abs_error))

###convert to numpy

temp = np.asarray(abs_error)

temp_error = []

for x in range(0,len(abs_error)-1):
    if not (-0.1 < abs_error[x] < 0.1):
        temp_error.append(x)

print(len(temp_error))


#print first 100 errors
print("Error Index  Abs error  TF Output  Kernel Out")

for x in range(0,100):
    try:
        print(temp_error[x] , abs_error[temp_error[x]] , tf_output[temp_error[x]] , kernel_output[temp_error[x]])
    except:
        print(x, " errors only.")
        break
