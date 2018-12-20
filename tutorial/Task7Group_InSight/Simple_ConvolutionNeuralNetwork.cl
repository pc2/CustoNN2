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

#define G_NUMBER_OF_IMAGES 10000
#define G_NUMBER_OF_FILERS 32
#define G_NUMBER_OF_IMAGE_ROWS 32
#define G_NUMBER_OF_IMAGE_COLS 32
#define G_NUMBER_OF_FILTER_ROWS 5
#define G_NUMBER_OF_FILTER_COLS 5
#define G_NUMBER_OF_CONV_OUT_ROWS 28
#define G_NUMBER_OF_CONV_OUT_COLS 28
#define G_MAXPOOL_OUT_ROWS 14
#define G_MAXPOOL_OUT_COLS 14
#define SR 8

channel int convOutChannel __attribute__((depth(32*32)));
channel int MaxPoolOutChannel __attribute__((depth(14*14)));

__kernel void ConvLayer(__global unsigned char * restrict img,__constant short * restrict cnnWeight,__constant short * restrict cnnBias,
                        const int numberOfImages,const int numberOfFilters,const int imgRows,const int imgCols,const int convFilterRows,const int convFilterCols,const int convOutRows,const int convOutCols)
{
        __local short cnnWeightLocal[G_NUMBER_OF_FILTER_ROWS*G_NUMBER_OF_FILTER_COLS*G_NUMBER_OF_FILERS];
        __local short cnnBiasLocal[G_NUMBER_OF_FILERS];

        //Load the weights into local memory
        #pragma unroll G_NUMBER_OF_FILERS
        for(int i=0; i<G_NUMBER_OF_FILTER_ROWS*G_NUMBER_OF_FILTER_COLS*G_NUMBER_OF_FILERS; i++)
                cnnWeightLocal[i]=cnnWeight[i];

        //Load the weights into local memory
        #pragma unroll
        for(int i=0; i<G_NUMBER_OF_FILERS; i++)
                cnnBiasLocal[i]=cnnBias[i];

        int numberOfTotalPixels = numberOfImages*imgRows*imgCols;
        int numberOfImagePixels = imgRows*imgCols;
        //printf("Conv Output\n");
        //For 10k images
        for(int imgIndex=0; imgIndex<G_NUMBER_OF_IMAGES; imgIndex++) {
              //  if(imgIndex%1000==0 || imgIndex==numberOfImages-1)
                    //    printf("Convolution for Image %d\n",imgIndex);

                int inX,inY=0;

                //for 32 filters
                for(int filterNumber=0; filterNumber<G_NUMBER_OF_FILERS; filterNumber++) {
                        //if(imgIndex==0)
                        //printf("For Filter %d\n",filterNumber);

                        //Conv Logic
                        for(int outRowIndex=0; outRowIndex<G_NUMBER_OF_CONV_OUT_ROWS; outRowIndex++) {
                                //printf("For outRowIndex %d\n",outRowIndex);
                                for(int outColIndex=0; outColIndex<G_NUMBER_OF_CONV_OUT_COLS; outColIndex++) {
                                        //printf("For outColIndex %d\n",outColIndex);
                                        //For Input indexing
                                        int conv=0;
                                        inX = outRowIndex;
                                        inY = outColIndex;
                                        conv = cnnBiasLocal[filterNumber]; //cnnBias
                                        //Filter
                                        #pragma unroll
                                        for(int filterRowIndex=0; filterRowIndex<G_NUMBER_OF_FILTER_ROWS; filterRowIndex++) {
                                                //Index for the Conv Filter
                                                int ConvFilterRowIndex = (filterNumber*G_NUMBER_OF_FILTER_ROWS*G_NUMBER_OF_FILTER_COLS)+(filterRowIndex*G_NUMBER_OF_FILTER_COLS);
                                                //Index for the Image
                                                int ConvImgRowIndex = (imgIndex*G_NUMBER_OF_IMAGE_ROWS*G_NUMBER_OF_IMAGE_COLS)+(inX*G_NUMBER_OF_IMAGE_COLS)+inY;


                                                conv+=cnnWeightLocal[ConvFilterRowIndex] * img[ConvImgRowIndex]
                                                       + cnnWeightLocal[ConvFilterRowIndex+1] * img[ConvImgRowIndex+1]
                                                       + cnnWeightLocal[ConvFilterRowIndex+2] * img[ConvImgRowIndex+2]
                                                       + cnnWeightLocal[ConvFilterRowIndex+3] * img[ConvImgRowIndex+3]
                                                       + cnnWeightLocal[ConvFilterRowIndex+4] * img[ConvImgRowIndex+4];
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
        __local int img[G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_IMAGE_COLS*G_NUMBER_OF_FILERS];
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

                                        p1 = img[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y)];
                                        p2 = img[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y+1)];
                                        p3 = img[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y+convOutCols)];
                                        p4 = img[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y+convOutCols+1)];
                                        m1 = max(p1,p2);
                                        m2 = max(p3,p4);
                                        currvalue= max(m1,m2);
                                        //Insert the max value in the channel
                                        write_channel_intel(MaxPoolOutChannel,currvalue);
                                        //  if(i==0)
                                        //  printf("%d  ",currvalue);
                                        currvalue=0;
                                }
                                //  if(i==0)
                                //  printf("\n ");
                        }
                        //  if(i==0)
                        //  printf("\n\n\n ");
                }

        }
}

/*
 * Kernel for Fully Connected Layer in CNN.
 * Input : 14*14*32 pixels for 1 Image. 10K Images in total
 * Input : 14*14*32 pixels for 1 Digit/Class. 10 Classes in Total.
 * Output : 1 class for each 10K images
 */
__kernel void FCLayer(__constant short * restrict digitWeights,const int numberOfFCPixels,const int NUMBER_OF_CLASSES,const int NUMBER_OF_IMAGES, __global int *  restrict kernelcalculatedLabels)
{

        int maxScore=0;
        int neuron=0;
        int score=0;
        int sumo[SR];

        __local int maxpooldata[6272];

        __local short digitWeightsLocal[G_MAXPOOL_OUT_ROWS*G_MAXPOOL_OUT_COLS*G_NUMBER_OF_FILERS*10];

        //Load weights into local memory
        for(int i=0; i<G_MAXPOOL_OUT_ROWS*G_MAXPOOL_OUT_COLS*G_NUMBER_OF_FILERS*10; i++)
                digitWeightsLocal[i]=digitWeights[i];

        //printf("FC Output\n");
        for(int count=0; count<G_NUMBER_OF_IMAGES; count++)
        {
                neuron=100; // Assigning some dummy digit class
                maxScore=0;

                //Store the Channels data of 1 Image in a linear array.
                for(int i=0; i<numberOfFCPixels; i++)
                        maxpooldata[i] = read_channel_intel(MaxPoolOutChannel);


                for(int weightIndex=0; weightIndex<NUMBER_OF_CLASSES; weightIndex++)
                {
                        #pragma unroll
                        for(int j=0; j<SR; j++)
                                sumo[j]=0;


                        score=0;
                        int sum =0;
                        #pragma unroll 32
                        for(int i=0; i<numberOfFCPixels; i++)
                        {
                                int temp;
                                //sum +=maxpooldata[i]*digitWeightsLocal[(weightIndex*numberOfFCPixels)+i];
                                temp =sumo[SR-1]+ (maxpooldata[i]*digitWeightsLocal[(weightIndex*numberOfFCPixels)+i]);
                                #pragma unroll
                                for(int k=SR-1; k>0; k--)
                                        sumo[k]=sumo[k-1];

                                sumo[0]=temp;

                        }
                        #pragma unroll
                        for(int l=0; l<SR; l++)
                                sum+=sumo[l];



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
