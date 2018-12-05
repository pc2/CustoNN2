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


__kernel void ConvLayer(__global unsigned char * restrict img,__global short * restrict cnnWeight,__global short * restrict cnnBias,__global int * restrict ConvOutput,
                        int numberOfImages,int numberOfFilters,int imgRows,int imgCols,int convFilterRows,int convFilterCols,int convOutRows,int convOutCols)
{
        int numberOfTotalPixels = numberOfImages*imgRows*imgCols;
        int numberOfImagePixels = imgRows*imgCols;
        //for(int imgIndex=0;imgIndex<numberOfTotalPixels;imgIndex+=numberOfImagePixels){

        int inX,inY=0;
        int conv=0;
        //Conv Logic
        for(int outRowIndex=0; outRowIndex<convOutRows; outRowIndex++) {
                for(int outColIndex=0; outColIndex<convOutCols; outColIndex++) {
                        //For Input indexing
                        inX = outRowIndex;
                        inY = outColIndex;
                        conv=0; //cnnBias
                        //Filter
                        for(int filterRowIndex=0; filterRowIndex<convFilterRows; filterRowIndex++) {
                                for(int filterColIndex=0; filterColIndex<convFilterCols; filterColIndex++) {
                                        conv+= cnnWeight[(filterRowIndex*convFilterRows)+filterColIndex] * img[(inX*imgRows)+inY];
                                        inY++;
                                }
                        }
                        //Next Row
                        inX++;
                        //reset Cols
                        inY=outColIndex;

                        // RELU
                        conv = conv>0 ? conv : 0;

                        ConvOutput[(outRowIndex*convOutRows)+outColIndex]=conv;
                        conv=0;
                }
        }

}
