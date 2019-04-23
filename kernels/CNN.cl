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






__kernel void ConvLayer(__global unsigned char * restrict img,__constant short * restrict cnnWeight,__constant short * restrict cnnBias,int G_NUMBER_OF_FILTER_ROWS,int G_NUMBER_OF_FILTER_COLS,int G_NUMBER_OF_FILERS,int G_NUMBER_OF_IMAGES,int G_NUMBER_OF_IMAGE_COLS,int G_NUMBER_OF_IMAGE_ROWS,int G_NUMBER_OF_CONV_OUT_ROWS, int G_NUMBER_OF_CONV_OUT_COLS,__global unsigned int * restrict output)
{
      	int m=0;

        int numberOfTotalPixels = G_NUMBER_OF_IMAGES*G_NUMBER_OF_IMAGE_ROWS*G_NUMBER_OF_IMAGE_COLS;
        int numberOfImagePixels = G_NUMBER_OF_IMAGE_ROWS*G_NUMBER_OF_IMAGE_COLS;
        //printf("Conv Output\n");
        //For 10k images
        for(int imgIndex=0; imgIndex<G_NUMBER_OF_IMAGES; imgIndex++) {
                //if(imgIndex%1000==0 || imgIndex==G_NUMBER_OF_IMAGES-1)
                      //  printf("Convolution for Image %d\n",imgIndex);

                int inX,inY=0;

                //for 32 filters
                for(int filterNumber=0; filterNumber<G_NUMBER_OF_FILERS; filterNumber++) {

                        //if(imgIndex==0)
                        //printf("For Filter %d\n",filterNumber);

                        //Conv Logic
                        for(int outRowIndex=0; outRowIndex<G_NUMBER_OF_CONV_OUT_ROWS; outRowIndex++) {
                                //printf("For outRowIndex %d\n",outRowIndex);
                          
                                #pragma unroll 28
                                for(int outColIndex=0; outColIndex<G_NUMBER_OF_CONV_OUT_COLS; outColIndex++) {

                                        //printf("For outColIndex %d\n",outColIndex);
                                        //For Input indexing
                                        int conv=0;
                                        inX = outRowIndex;
                                        inY = outColIndex;
                                        conv = cnnBias[filterNumber]; //cnnBias
                                        //Filter
                                        #pragma unroll 5
                                        for(int filterRowIndex=0; filterRowIndex<G_NUMBER_OF_FILTER_ROWS; filterRowIndex++) {
                                                //Index for the Conv Filter
                                                int ConvFilterRowIndex = (filterNumber*G_NUMBER_OF_FILTER_ROWS*G_NUMBER_OF_FILTER_COLS)+(filterRowIndex*G_NUMBER_OF_FILTER_COLS);
                                                //Index for the Image
                                                int ConvImgRowIndex = (imgIndex*G_NUMBER_OF_IMAGE_ROWS*G_NUMBER_OF_IMAGE_COLS)+(inX*G_NUMBER_OF_IMAGE_COLS)+inY;


                                                conv+=cnnWeight[ConvFilterRowIndex] * img[ConvImgRowIndex]
                                                       + cnnWeight[ConvFilterRowIndex+1] * img[ConvImgRowIndex+1]
                                                       + cnnWeight[ConvFilterRowIndex+2] * img[ConvImgRowIndex+2]
                                                       + cnnWeight[ConvFilterRowIndex+3] * img[ConvImgRowIndex+3]
                                                       + cnnWeight[ConvFilterRowIndex+4] * img[ConvImgRowIndex+4];
                                                //Next Row
                                                inX++;
                                                //reset Cols
                                                inY=outColIndex;
                                        }

                                        // RELU
                                        conv = conv>0 ? conv : 0;

                                        output[m] = conv;
					m++;
                                        //if(imgIndex==0)
                                        //  printf("%d  ",conv);
                                        //ConvOutput[(imgIndex*numberOfFilters*convOutRows*convOutCols)+(filterNumber*convOutRows*convOutCols)+(outRowIndex*convOutRows)+outColIndex]=conv;
                                        //write_channel_intel(convOutChannel,conv);
                                        conv=0;
                                }
                                //if(imgIndex==0)
                                //  printf("\n");
                                
                        }
                        //  if(imgIndex==0)
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
__kernel void MaxPool( __global int * restrict ConvOutput,int G_NUMBER_OF_CONV_OUT_COLS,int G_NUMBER_OF_CONV_OUT_ROWS,int G_NUMBER_OF_FILERS,int G_MAXPOOL_STRIDE,int G_NUMBER_OF_IMAGES,int G_NUMBER_OF_IMAGE_COLS,__global unsigned int * restrict Maxoutput)
{

        //struct conv_buffer conv1;
        //  printf("Maxpool Output\n");
	int m=0;
        for ( int i =0; i < G_NUMBER_OF_IMAGES; ++i)
        {

                int currvalue=0;



                for (int k = 0; k <G_NUMBER_OF_FILERS; ++k)
                {
                  
                        //Store the Channels data of 1 Image in a linear array.


                        //conv1=read_channel_intel(convOutChannel);
                   

                        
                        for (int x = 0; x < G_NUMBER_OF_CONV_OUT_ROWS; x=x+G_MAXPOOL_STRIDE)
                        {
                               
                                #pragma unroll 14
                                for (int y = 0; y < G_NUMBER_OF_CONV_OUT_COLS; y=y+G_MAXPOOL_STRIDE)
                                {


                                        int p1,p2,p3,p4,m1,m2;
                                        p1 = ConvOutput[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y)];
                                        p2 = ConvOutput[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y+1)];
                                        p3 = ConvOutput[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y+G_NUMBER_OF_CONV_OUT_COLS)];
                                        p4 = ConvOutput[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y+G_NUMBER_OF_CONV_OUT_COLS+1)];
                                        m1 = max(p1,p2);
                                        m2 = max(p3,p4);
                                        currvalue= max(m1,m2);
                                        //Insert the max value in the channel
                                        //write_channel_intel(MaxPoolOutChannel,currvalue);
                                        Maxoutput[m]=currvalue;
                                        m++;
                                        //if(i==0 )
                                        //printf("%d  ",currvalue);

                                        currvalue=0;
                                }

                                //if(i==0 )
                                //  printf("\n ");
                                
					
                        //if(i==0)
                        //("\n\n\n ");


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
__kernel void FCLayer(__global unsigned int * restrict Maxoutput,__constant short * restrict digitWeights,const int numberOfFCPixels,const int NUMBER_OF_CLASSES,const int NUMBER_OF_IMAGES,__global int * restrict
kernelcalculatedLabels,int G_MAXPOOL_OUT_ROWS,int G_MAXPOOL_OUT_COLS,int G_NUMBER_OF_FILERS,__global unsigned int * restrict FCLoutput,const int SR)
{

        // int maxScore=0;
        // int neuron=0;
        // int score=0;
        int sumo[8];

        //printf("FC Output\n");
        for(int count=0; count<NUMBER_OF_IMAGES; count++)
        {
                int neuron=100; // Assigning some dummy digit class
                int maxScore=0;
                int maxpooldata[6272];
                //Store the Channels data of 1 Image in a linear array.
                for(int i=0; i<G_NUMBER_OF_FILERS; i++) {

                        for(int q=0; q<G_MAXPOOL_OUT_ROWS; q++) {
                          
                          #pragma unroll 14
                            for(int l=0;l<G_MAXPOOL_OUT_COLS;l++)
                                maxpooldata[(i*G_MAXPOOL_OUT_ROWS*G_MAXPOOL_OUT_COLS)+(q*G_MAXPOOL_OUT_ROWS)+l] = Maxoutput[l];
                        }
                }


                for(int weightIndex=0; weightIndex<NUMBER_OF_CLASSES; weightIndex++)
                {
                        #pragma unroll
                        for(int j=0; j<SR; j++)
                                sumo[j]=0;


                        int score=0;
                        int sum =0;
                        #pragma unroll 32
                        for(int i=0; i<numberOfFCPixels; i++)
                        {
                                int temp;
                                //sum +=maxpooldata[i]*digitWeightsLocal[(weightIndex*numberOfFCPixels)+i];
                                temp =sumo[SR-1]+ (maxpooldata[i]*digitWeights[(weightIndex*numberOfFCPixels)+i]);
                                #pragma unroll 
                                for(int k=SR-1; k>0; k--)
                                        sumo[k]=sumo[k-1];

                                sumo[0]=temp;

                        }
                        #pragma unroll 8
                        for(int l=0; l<SR; l++)
                                sum+=sumo[l];

                        score=sum;
                        //if(count==4500)
                        //printf(" 45 --  %d -- %d\n  ",weightIndex,score);

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

