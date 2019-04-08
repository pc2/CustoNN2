// Function for the Convolution Layer
__kernel void ConvolutionLayer(__global double * restrict img,  
				__global double * restrict output, 
				__global double * restrict filter, 
				int number_of_filters, 
				int number_of_image_rows, 
				int number_of_image_cols, 
				int conv_stride){
	
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;

	index = 0;
	image_size = number_of_image_rows * number_of_image_cols;
	for (layer = 1; layer <=number_of_filters; layer++) {
		image_index = 0;
		for (i = 0; i < number_of_image_rows; i++) {
			for (j = 0; j < number_of_image_cols; j++) {
				current_layer = layer * 26;
				output[index] = filter[current_layer - 1];
				filter_index = 0;
				for (k = - conv_stride; k <= conv_stride; k++) {
					for (t = - conv_stride; t <= conv_stride; t++) {
						temp_index = 28 * k + t + image_index;
						if ((j == 0 && t < 0) || (j == number_of_image_cols-1 && t > 0)) {
							temp_index = -1;
						}
						if ((j == 1 && t < -1) || (j == number_of_image_cols-2 && t > 1)) {
							temp_index = -1;
						}
						if ((i == 0 && k < 0) || (i == number_of_image_rows-1 && k > 0)) {
							temp_index = -1;
						}
						if ((i == 1 && k < -1) || (i == number_of_image_rows-2 && k > 1)) {
							temp_index = -1;
						}
						if ((temp_index >= 0) && (temp_index < image_size)) {
							output[index] += img[temp_index]*filter[filter_index+current_layer-26];
						}
						filter_index++;
					}
				}
				if (output[index] < (double)0)
					output[index] = (double)0;
				index++;
				image_index++;

			}
		}
	}	
}
