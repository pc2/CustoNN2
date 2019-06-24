/*
 * Kernel for Avgpooling Layer
 * Used for reducing the dimension of images.
 */

__kernel void AvgPool(__global int * restrict input, 
			int number_of_image_rows, 
			int  number_of_image_cols,
			int number_of_filters,
			int kernel_size, 
			int stride,
			int number_of_images,
			__global int * restrict output){
	double avgpool[200]; 
	int image_size = number_of_image_rows * number_of_image_cols * number_of_filters * number_of_images;
	int avg=0, i, oindex=0,count=kernel_size, s=kernel_size, k, f, startIndex=0, endIndex=s, imageIndex=1, j=0;

   	for(f=0;f<s;f++){
	   while(count!=0){
		for(k=startIndex;k<endIndex;k++){
		    avgpool[j]=input[k];
		    j++;  
		}
		count--; 
		startIndex=number_of_image_cols*imageIndex;
		imageIndex++;       
		endIndex=startIndex+s;       
	    }
   	}
	for(i=0;i<s*s;i++){
       		avg=avg+avgpool[i];
   	}
	avg=avg/(s*s);
	output[oindex]=avg;
	oindex++;
}





