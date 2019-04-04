#define G_NUMBER_OF_IMAGES 10000
#define G_NUMBER_OF_FILTERS 32
#define G_NUMBER_OF_IMAGE_ROWS 28
#define G_NUMBER_OF_IMAGE_COLS 28
#define G_NUMBER_OF_FILTER_ROWS 5
#define G_NUMBER_OF_FILTER_COLS 5
#define G_CONV_STRIDE 2

// Function for the Convolution Layer
__kernel void ConvolutionLayer(__global double * restrict img,  
				__global double * restrict output, 
				__global double * restrict filter){
	
	int i,j,k,t;
	int index, temp_index, filter_index, image_index;
	int layer, current_layer, image_size;

	index = 0;
	image_size = G_NUMBER_OF_IMAGE_ROWS * G_NUMBER_OF_IMAGE_COLS;
	for (layer = 1; layer <=G_NUMBER_OF_FILTERS; layer++) {
		image_index = 0;
		for (i = 0; i < G_NUMBER_OF_IMAGE_ROWS; i++) {
			for (j = 0; j < G_NUMBER_OF_IMAGE_COLS; j++) {
				current_layer = layer * 26;
				output[index] = filter[current_layer - 1];
				filter_index = 0;
				for (k = - G_CONV_STRIDE; k <= G_CONV_STRIDE; k++) {
					for (t = - G_CONV_STRIDE; t <= G_CONV_STRIDE; t++) {
						temp_index = 28 * k + t + image_index;
						if ((j == 0 && t < 0) || (j == G_NUMBER_OF_IMAGE_COLS-1 && t > 0)) {
							temp_index = -1;
						}
						if ((j == 1 && t < -1) || (j == G_NUMBER_OF_IMAGE_COLS-2 && t > 1)) {
							temp_index = -1;
						}
						if ((i == 0 && k < 0) || (i == G_NUMBER_OF_IMAGE_ROWS-1 && k > 0)) {
							temp_index = -1;
						}
						if ((i == 1 && k < -1) || (i == G_NUMBER_OF_IMAGE_ROWS-2 && k > 1)) {
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