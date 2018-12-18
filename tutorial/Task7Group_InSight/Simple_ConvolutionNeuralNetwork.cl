/*
 * Kernel for Convolution Layer in CNN
 * img : 1D vector having 10k*32*32  elements
 * cnnWeight : 1D vector having 32*28*28 elements
 * cnnBias : Bias weight for 32 filters. 1D vector having 32 elements.
 * numberOfImages : Number of images in the dataset =10k
 * numberOfFilters : number of convolution filters = 32
 * imgRows : Number of rows in the input image
 * imgCols : NUmber of cols in the input images
 * convFilterRows : NUmber of Rows in the Conv Filter
 * convFilterCols : NUmber of Cols in the Conv Filter
 * convOutRows : Number of rows in the output image
 * convOutCols : Number of Cols in the output image
 * Output : 32*28*28 image will be transferred to MaxPool using channel
 */
//Enable the channel extension
 #pragma OPENCL EXTENSION cl_intel_channels : enable

channel int convOutChannel __attribute__((depth(0)));
channel int MaxPoolOutChannel __attribute__((depth(0)));

__kernel void ConvLayer(__global unsigned char * restrict img,__global short * restrict cnnWeight,__global short * restrict cnnBias,
                        int numberOfImages,int numberOfFilters,int imgRows,int imgCols,int convFilterRows,int convFilterCols,int convOutRows,int convOutCols)
{
        int numberOfTotalPixels = numberOfImages*imgRows*imgCols;
        int numberOfImagePixels = imgRows*imgCols;
        //printf("Conv Output\n");
        //For 10k images
        for(int imgIndex=0; imgIndex<numberOfImages; imgIndex++) {
                //if(imgIndex%1000==0 || imgIndex==numberOfImages-1)
                        //printf("Convolution for Image %d\n",imgIndex);

                int inX,inY=0;
                int conv=0;
                //for 32 filters
                for(int filterNumber=0; filterNumber<numberOfFilters; filterNumber++) {
                  //  if(imgIndex==0)
                      //printf("For Filter %d\n",filterNumber);
                        //Conv Logic
                        for(int outRowIndex=0; outRowIndex<convOutRows; outRowIndex++) {
                                //printf("For outRowIndex %d\n",outRowIndex);
                                for(int outColIndex=0; outColIndex<convOutCols; outColIndex++) {
                                        //printf("For outColIndex %d\n",outColIndex);
                                        //For Input indexing
                                        inX = outRowIndex;
                                        inY = outColIndex;
                                        conv = cnnBias[filterNumber]; //cnnBias
                                        //Filter
					//#pragma unroll 5
                                        for(int filterRowIndex=0; filterRowIndex<convFilterRows; filterRowIndex++) {
                                                //printf("\t For filterRowIndex %d\n",filterRowIndex);
						//#pragma unroll 5
                                                for(int filterColIndex=0; filterColIndex<convFilterCols; filterColIndex++) {
                                                        //printf("\t\t For filterColIndex %d\n",filterColIndex);
                                                        //printf("Img:%d \n ",img[(imgIndex*numberOfImages)+(inX*imgRows)+inY]);
                                                        //printf("Conv:%d \n", cnnWeight[(filterNumber*numberOfFilters)+(filterRowIndex*convFilterRows)+filterColIndex]);
                                                        conv+= cnnWeight[(filterNumber*convFilterRows*convFilterCols)+(filterRowIndex*convFilterRows)+filterColIndex] * img[(imgIndex*imgRows*imgCols)+(inX*imgRows)+inY];
                                                        inY++;
                                                }
                                                //Next Row
                                                inX++;
                                                //reset Cols
                                                inY=outColIndex;
                                        }


                                        // RELU
                                        conv = conv>0 ? conv : 0;
                                        //if(imgIndex==0)
                                          //  printf("%d  ",conv);
                                        //ConvOutput[(imgIndex*numberOfFilters*convOutRows*convOutCols)+(filterNumber*convOutRows*convOutCols)+(outRowIndex*convOutRows)+outColIndex]=conv;
                                        write_channel_intel(convOutChannel,conv);
                                        conv=0;
                                }
                                  //if(imgIndex==0)
                                  //printf("\n");
                        }
                        //if(imgIndex==0)
                        //printf("\n\n");
                }
        }

}

/*
 * Kernel for Maxpool Layer in CNN
 * This Kernel will downsample the image output from the convolution layer. We are selecting the Max value of the 4 pixels.
 * Stride : 2
 * Input : 28*28*32 Pixels for 1 Image. 10k Images in total
 * Output : 14*14*32 Pixels for 1 Image, 10K Images in total sent through Channel to FC.
 */
__kernel void MaxPool(int numberOfImages, int numberOfFilters,int convOutRows,int convOutCols,int stride)
{
        int currvalue=0;
        int p1,p2,p3,p4,m1,m2;
        int img[30000];
        //  printf("Maxpool Output\n");
        for ( int i =0; i < numberOfImages; ++i)
        {
                //Store the Channels data of 1 Image in a linear array.
                for ( int j = 0; j<numberOfFilters*convOutRows*convOutCols; j++ )
                        img[j] = read_channel_intel(convOutChannel);


                for (int k = 0; k <numberOfFilters; ++k)
                {
                        for (int x = 0; x < convOutRows; x=x+stride)
                        {	
                                for (int y = 0; y < convOutCols; y=y+stride)
                                {

                                        p1 = img[(k*convOutRows*convOutCols)+(x*convOutCols)+(y)];
                                        p2 = img[(k*convOutRows*convOutCols)+(x*convOutCols)+(y+1)];
                                        p3 = img[(k*convOutRows*convOutCols)+(x*convOutCols)+(y+convOutCols)];
                                        p4 = img[(k*convOutRows*convOutCols)+(x*convOutCols)+(y+convOutCols+1)];
                                        m1 = max(p1,p2);
                                        m2 = max(p3,p4);
                                        currvalue= max(m1,m2);
                                        //Insert the max value in the channel
                                        write_channel_intel(MaxPoolOutChannel,currvalue);
                                        //  printf("%d  ",currvalue);
                                        currvalue=0;
                                }
                                //    printf("\n ");
                        }

                }

        }
}

/*
 * Kernel for Fully Connected Layer in CNN.
 * Input : 14*14*32 pixels for 1 Image. 10K Images in total
 * Input : 14*14*32 pixels for 1 Digit/Class. 10 Classes in Total.
 * Output : 1 class for each 10K images
 */
__kernel void FCLayer(__global short * restrict digitWeights,int numberOfFCPixels, int NUMBER_OF_CLASSES, int NUMBER_OF_IMAGES, __global int *  restrict kernelcalculatedLabels)
{

        int maxScore=0;
        int neuron=0;
        int score=0;
	int sumo[34];

        int maxpooldata[6272];
	
	

        //printf("FC Output\n");
        for(int count=0; count<NUMBER_OF_IMAGES; count++)
        {
                neuron=100;
                maxScore=0;
		
                //Store the Channels data of 1 Image in a linear array.
                for(int i=0; i<numberOfFCPixels; i++)
                        maxpooldata[i] = read_channel_intel(MaxPoolOutChannel);


                for(int weightIndex=0; weightIndex<NUMBER_OF_CLASSES; weightIndex++)
                {
			for(int j=0;j<34;j++)
		{
			sumo[j]=0;
		}
		

                        score=0;
                        int sum =0;
			#pragma unroll 64
                        for(int i=0; i<numberOfFCPixels; i++)
                        {
				int temp;
                                //sum +=maxpooldata[i]*digitWeights[(weightIndex*numberOfFCPixels)+i];
				temp =sumo[33]+ (maxpooldata[i]*digitWeights[(weightIndex*numberOfFCPixels)+i]);
				for(int k=34;k>0;k--)
				{
					sumo[k]=sumo[k-1];
				}
			sumo[0]=temp;
                        }
			for(int l=0;l<34;l++)
				{
					sum+=sumo[l];
				}

                        score=sum;
                        //if(count==0)
                          //printf("%d -- %d\n  ",weightIndex,score);

                        // Max Score logic
                        if(score>maxScore)
                        {
                                maxScore=score;
                                neuron=weightIndex;
                        }
                }
                kernelcalculatedLabels[count]=neuron;

        }
}
