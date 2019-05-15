/*
 * Kernel for Avgpooling Layer
 * Used for reducing the dimension of images. We use a 5*5 matrix for avgpooling.
 */

__kernel void AvgPool5_5(__global int * restrict input, 
			int number_of_image_rows, 
			int  number_of_image_cols,
			int number_of_filters, 
			int stride,
			int number_of_images,
			__global int * restrict output){
	double avgpool[25]; 
	int image_size = number_of_image_rows * number_of_image_cols * number_of_filters * number_of_images;
	int avg=0, i, input[300], count=5, s=5, k, f, startIndex=0, endIndex=s, imageIndex=1, j=0;

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
    	return avg;
}


/*
 * Kernel for Avgpooling Layer
 * Used for reducing the dimension of images. We use a 7*7 matrix for avgpooling.
 */

__kernel void AvgPool7_7(__global int * restrict input, 
			int number_of_image_rows, 
			int  number_of_image_cols,
			int number_of_filters, 
			int stride,
			int number_of_images,
			__global int * restrict output){

	int image_size = number_of_image_rows * number_of_image_cols * number_of_filters * number_of_images;
	int avg=0, i, count=7, s=7, k, f, startIndex=0, endIndex=s, imageIndex=1, j=0;

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
    	return avg;
}



