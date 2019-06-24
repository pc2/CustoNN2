
__kernel void ConcatLayer(__global double * restrict input_1, 
			  __global double * restrict input_2, 
			  __global double * restrict input_3, 
			  __global double * restrict input_4, 
			  int input_1_rows, int input_1_cols, 
			  int input_1_filters, 
			  int input_2_rows, int input_2_cols, 
			  int input_2_filters, 
			  int input_3_rows, int input_3_cols, 
			  int input_3_filters, 
			  int input_4_rows, int input_4_cols, 
			  int input_4_filters, 
			  __global double * restrict output){
	int image_layers = 1; //Since image is RGB we have 3 layers for each image
	int total_inputs = 4;
	//input_1  convolution layer input 10*10*3*4 where 4 is the number pf filters
	
	int temp_list[4][4]; 
	//to store dimensions [][0] 28x28 and [][1] filter number
	temp_list[0][0] = input_1_rows;
	temp_list[1][0] = input_2_rows;
	temp_list[2][0] = input_3_rows;
	temp_list[3][0] = input_4_rows;
	temp_list[0][1] = input_1_cols;
	temp_list[1][1] = input_2_cols;
	temp_list[2][1] = input_3_cols;
	temp_list[3][1] = input_4_cols;
	temp_list[0][2] = input_1_filters;
	temp_list[1][2] = input_2_filters;
	temp_list[2][2] = input_3_filters;
	temp_list[3][2] = input_4_filters;
	temp_list[0][3] = total_inputs;
	temp_list[1][3] = total_inputs;
	temp_list[2][3] = total_inputs;
	temp_list[3][3] = total_inputs;
	
	printf("Entered concat\n");
	int count = 0, total_range = 0, range = 0;
	//total_range is used for adding different depths inputs eg input_1 =108 in[ut_2=208 so for input_3 to be concatenated properly in output. total_range = 108+208
	for(int i = 1; i<=total_inputs; i++){
		double temp_var[108];
		if(i==1){
			range = temp_list[(i-1)][0]*temp_list[(i-1)][1]*temp_list[(i-1)][2]*temp_list[(i-1)][3];
			//rowsxcolsxfiltersxtotalimagesperarray
			for(int x = 1; x<=range; x++){
				temp_var[x-1] = input_1[x-1];
				count+=1;
				
			}
		}
		if(i==2){
			range = temp_list[(i-1)][0]*temp_list[(i-1)][1]*temp_list[(i-1)][2]*temp_list[(i-1)][3];
			//rowsxcolsxfiltersxtotalimagesperarray
			for(int x = 1; x<=range; x++){
					temp_var[x-1] = input_2[x-1];
			}
		}
		if(i==3){
			range = temp_list[(i-1)][0]*temp_list[(i-1)][1]*temp_list[(i-1)][2]*temp_list[(i-1)][3];
			//rowsxcolsxfiltersxtotalimagesperarray
			for(int x = 1; x<=range; x++){
					temp_var[x-1] = input_3[x-1];
			}
		}
		if(i==4){
			range = temp_list[(i-1)][0]*temp_list[(i-1)][1]*temp_list[(i-1)][2]*temp_list[(i-1)][3];
			//rowsxcolsxfiltersxtotalimagesperarray
			for(int x = 1; x<=range; x++){
					temp_var[x-1] = input_4[x-1];
			}
		}


		for(int filter=0; filter<=range; filter++){

			output[total_range+filter]+=temp_var[filter];
		}
		total_range+=range;
		//append old range to total range

	}
	


}
