## GoogLeNet kernels

# Convolution kernel.
Variables:


We need to consider this things:
- (28,28,64) -> (28,28,128)
- diff padding (1,3)
- diff kernel sizes
- diff strides


# MaxPooling kernel. 
Variables:
- __global double * restrict input,
- int number_of_images,
- int number_of_image_rows,
- int number_of_image_cols,
- int number_of_filters,
- int stride,
- int padding,
- __global double * restrict output

Example: max(2,3,4,5) = 5

# AvgPooling kernel.
Variables:
- __global double * restrict input, 
- int number_of_images, 
- int number_of_image_rows, 
- int number_of_image_cols, 
- int number_of_filters, 
- int stride, int padding,
- __global double * restrict output

Example: avg(2,3,4,5) = 2,5

# SoftMax kernel.
Variables:
- __global double * restrict input, 
- int number_of_classes, 
- __global double * restrict output

For example see this link: https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax

# Concatincation kernel.
Variables:
- __global double * restrict input,
- int number_of_input_blocks,
- int * numbers_of_layers,
- int numbers_of_image_rows,
- int numbers_of_image_rows,
- __global double * restrict output

Example: (28,28,64), (28,28,24), (28,28,65) => (28,28, 64+24+65)
