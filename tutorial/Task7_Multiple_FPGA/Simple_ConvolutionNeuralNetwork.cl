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


//PreProcessor Statements
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
#define G_MAXPOOL_STRIDE 2
#define SR 66

//Struct to hold 1 Row Output of the Conv Layer
typedef struct conv_buffer {
        int temp_buffer[G_NUMBER_OF_CONV_OUT_COLS];
}co;

//Struct to hold 1 row Output of Maxpool Layer
typedef struct max_buffer {
        int maxPool_buffer[8];
}maxStruct;

//Channel Between Conv Layer and Maxpool
/*
channel co ConvOutChannel __attribute__((depth(64)))
                        __attribute__((io("kernel_output_ch0")));
channel co MaxPoolInChannel __attribute__((depth(64)))
                        __attribute__((io("kernel_output_ch0")));
*/
channel co ConvOutChannel __attribute__((depth(64)));                        
//Channel Between Maxpool and FC Layer
channel maxStruct MaxPoolOutChannel __attribute__((depth(0))) __attribute__((io("kernel_output_ch0"))); // Channel Tx
channel maxStruct FCInChannel __attribute__((depth(0))) __attribute__((io("kernel_input_ch0")));  // Channel Rx


__kernel void ConvLayer(__global unsigned char * restrict img,__constant short * restrict cnnWeight,__constant short * restrict cnnBias)
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

        int numberOfTotalPixels = G_NUMBER_OF_IMAGES*G_NUMBER_OF_IMAGE_ROWS*G_NUMBER_OF_IMAGE_COLS;
        int numberOfImagePixels = G_NUMBER_OF_IMAGE_ROWS*G_NUMBER_OF_IMAGE_COLS;
        //printf("Conv Output\n");
        //For 10k images
        for(int imgIndex=0; imgIndex<G_NUMBER_OF_IMAGES; imgIndex++) {
                //if(imgIndex%1000==0 || imgIndex>G_NUMBER_OF_IMAGES-100)
                 //       printf("Convolution for Image %d\n",imgIndex);

                int inX,inY=0;

                //for 32 filters
                for(int filterNumber=0; filterNumber<G_NUMBER_OF_FILERS; filterNumber++) {

                        //if(imgIndex==0)
                        //printf("For Filter %d\n",filterNumber);

                        //Conv Logic
                        
                        for(int outRowIndex=0; outRowIndex<G_NUMBER_OF_CONV_OUT_ROWS; outRowIndex++) {
                                //printf("For outRowIndex %d\n",outRowIndex);
                                struct conv_buffer co1;
                                #pragma unroll
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

                                        co1.temp_buffer[outColIndex] = conv;
                                        //if(imgIndex==0)
                                        //  printf("%d  ",conv);
                                        //ConvOutput[(imgIndex*numberOfFilters*convOutRows*convOutCols)+(filterNumber*convOutRows*convOutCols)+(outRowIndex*convOutRows)+outColIndex]=conv;
                                        //write_channel_intel(convOutChannel,conv);
                                        conv=0;
                                }
                                //if(imgIndex==0)
                                //  printf("\n");
                                write_channel_intel(ConvOutChannel,co1);
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
__kernel void MaxPool()
{

        //struct conv_buffer conv1;
         //printf("Maxpool Output\n");
        for ( int i =0; i < G_NUMBER_OF_IMAGES; ++i)
        {

                int currvalue=0;

                //if(i%1000==0 || i>G_NUMBER_OF_IMAGES-100)
                //        printf("Maxpool for Image %d\n",i);
                for (int k = 0; k <G_NUMBER_OF_FILERS; ++k)
                {
                  int img[G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_IMAGE_COLS*G_NUMBER_OF_FILERS];
                        //Store the Channels data 5of 1 Image in a linear array.
                        
                        for ( int j = 0; j<G_NUMBER_OF_CONV_OUT_ROWS; j++ ) {
                                struct conv_buffer conv1 = read_channel_intel(ConvOutChannel);
                                #pragma unroll
                                for(int l=0; l<G_NUMBER_OF_CONV_OUT_COLS; l++) {
                                        img[(k*G_NUMBER_OF_CONV_OUT_COLS*G_NUMBER_OF_CONV_OUT_ROWS)+(j*G_NUMBER_OF_CONV_OUT_COLS)+l]=conv1.temp_buffer[l];
                                }
                        }

                        //conv1=read_channel_intel(convOutChannel);
                        //struct max_buffer max1[G_NUMBER_OF_CONV_OUT_ROWS/G_MAXPOOL_STRIDE];
                        int bufferCount=0;
                        
                        for (int x = 0; x < G_NUMBER_OF_CONV_OUT_ROWS; x=x+G_MAXPOOL_STRIDE)
                        {
                                int m=0;
                                struct max_buffer max1; // 1st half of row ( 7 Pixels)
                                
                                for (int y = 0; y < G_NUMBER_OF_CONV_OUT_COLS; y=y+G_MAXPOOL_STRIDE)
                                {


                                        int p1,p2,p3,p4,m1,m2;
                                        p1 = img[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y)];
                                        p2 = img[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y+1)];
                                        p3 = img[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y+G_NUMBER_OF_CONV_OUT_COLS)];
                                        p4 = img[(k*G_NUMBER_OF_CONV_OUT_ROWS*G_NUMBER_OF_CONV_OUT_COLS)+(x*G_NUMBER_OF_CONV_OUT_COLS)+(y+G_NUMBER_OF_CONV_OUT_COLS+1)];
                                        m1 = max(p1,p2);
                                        m2 = max(p3,p4);
                                        currvalue= max(m1,m2);
                                        //Insert the max value in the channel
                                        //write_channel_intel(MaxPoolOutChannel,currvalue);
                                        max1.maxPool_buffer[m]=currvalue;     
                                        m++;
                                       
                                        if( m==G_MAXPOOL_OUT_COLS/2){
                                                m=0;
                                                write_channel_intel(MaxPoolOutChannel,max1);
                                        }        
                                        //if(i==0 )
                                        //printf("%d  ",currvalue);

                                        currvalue=0;
                                }

                                //if(i==0 )
                                //  printf("\n ");
                                //write_channel_intel(MaxPoolOutChannel,max1);
                               // write_channel_intel(MaxPoolOutChannel,max2);
                              //  bufferCount++;
                        }
                        //if(i==0)
                        //("\n\n\n ");


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

        // int maxScore=0;
        // int neuron=0;
        // int score=0;
        int sumo[SR];

        //__local int maxpooldata[6272];

        __local short digitWeightsLocal[G_MAXPOOL_OUT_ROWS*G_MAXPOOL_OUT_COLS*G_NUMBER_OF_FILERS*10];

        //Load weights into local memory
        for(int i=0; i<G_MAXPOOL_OUT_ROWS*G_MAXPOOL_OUT_COLS*G_NUMBER_OF_FILERS*10; i++)
                digitWeightsLocal[i]=digitWeights[i];

        //printf("FC Output\n");
        for(int count=0; count<G_NUMBER_OF_IMAGES; count++)
        {
                int neuron=100; // Assigning some dummy digit class
                int maxScore=0;
                int maxpooldata[6272];
                //Store the Channels data of 1 Image in a linear array.
                
                for(int i=0; i<G_NUMBER_OF_FILERS; i++) {
                        
                        for(int q=0; q<G_MAXPOOL_OUT_ROWS; q++) {
                                struct max_buffer max1[2];
                                for(int colData=0;colData<2;colData++){                                
                                        max1[colData]= read_channel_intel(FCInChannel);
                                }
                                #pragma unroll
                                for(int colData=0;colData<2;colData++){
                                        #pragma unroll
                                        for(int l=0;l<=G_MAXPOOL_OUT_COLS/2;l++){
                                                maxpooldata[(i*G_MAXPOOL_OUT_ROWS*G_MAXPOOL_OUT_COLS)+(q*G_MAXPOOL_OUT_ROWS)+(colData*(G_MAXPOOL_OUT_COLS/2))+l] = max1[colData].maxPool_buffer[l];
                                       }
                                }
                                
                        }
                }


                for(int weightIndex=0; weightIndex<NUMBER_OF_CLASSES; weightIndex++)
                {
                        #pragma unroll
                        for(int j=0; j<SR; j++)
                                sumo[j]=0;


                        int score=0;
                        int sum =0;
                        #pragma unroll 64
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
                        

                        // Max Score logic
                        if(score>maxScore)
                        {
                                maxScore=score;
                                neuron=weightIndex;
                        }
                }
                //if(count%1000==0 || count>G_NUMBER_OF_IMAGES-100 )
                //       printf(" FC %d --  %d \n  ",count,neuron);
                kernelcalculatedLabels[count]=neuron;

        }
}

