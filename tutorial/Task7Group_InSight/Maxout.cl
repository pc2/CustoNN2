channel int MaxPoolOutChannel __attribute__((depth(0)));
__kernel void MaxPool(__global int * restrict ConvOutput ,__global int * restrict MPOutput, int numberOfFilters,int imgRows,int imgCols, int stride )
{
int currvalue=0;
for (int k = 0; k <numberOfFilters ; ++k)
        {
	for (int x = 0; x < imgRows; x=x+stride)
                {
                        for (int y = 0; y < imgCols; y=y+stride)
                        {
                                for (int i = 0; i < stride; ++i)
                                {
                                        for (int j = 0; j < stride; ++j)
                                        {
                                                int updatevalue = read_channel_intel(convOutChannel);
                                                currvalue= max(currvalue, updatevalue);
                                        }
                                }
                                MPOutput[k][x/2][y/2] = currvalue;
		 write_channel_intel(MaxPoolOutChannel,currvalue);
                                currvalue=0;
                        }
                }
        }
}
