
__kernel void ConcatLayer(__global double * restrict input_1, 
			  __global double * restrict input_2, 
			  __global double * restrict input_3, 
			  __global double * restrict input_4, 
			  int input_1_dim, 
			  int input_1_filters, 
			  int input_2_dim, 
			  int input_2_filters, 
			  int input_3_dim, 
			  int input_3_filters, 
			  int input_4_dim, 
			  int input_4_filters, 
			  __global double * restrict output){

	int image_layers = 3; //Since image is RGB we have 3 layers for each image
	int total_inputs = 4;
	//input_1  convolution layer input 10*10*3*4 where 4 is the number pf filters
	
	
	int temp_list[4][2]; 
	//to store dimensions [][0] 28x28 and [][1] filter number
	temp_list[0][0] = input_1_dim;
	temp_list[1][0] = input_2_dim;
	temp_list[2][0] = input_3_dim;
	temp_list[3][0] = input_4_dim;
	temp_list[0][1] = input_1_filters;
	temp_list[1][1] = input_2_filters;
	temp_list[2][1] = input_3_filters;
	temp_list[3][1] = input_4_filters;
	

	for(int i = 1; i<=total_inputs; i++){
		double temp_var[100000];
		if(i==1){
			for(int x = 0; x<temp_list[(i-1)][0]; x++){
				temp_var[x] = input_1[x];
			}
		}
		if(i==2){
			for(int x = 0; x<temp_list[(i-1)][0]; x++){
				temp_var[x] = input_2[x];
			}
		}
		if(i==3){
			for(int x = 0; x<temp_list[(i-1)][0]; x++){
				temp_var[x] = input_3[x];
			}
		}
		if(i==4){
			for(int x = 0; x<temp_list[(i-1)][0]; x++){
				temp_var[x] = input_4[x];
			}
		}
		for(int filter=1; filter<=temp_list[(i-1)][1]; filter++){
			for(int layer=1; layer <= image_layers; layer++){		
				for(int c=1; c <= temp_list[(i-1)][0]; c++){
					for(int r=1; r <= temp_list[i][0]; r++){
					
						output[(i*r*c*layer*filter)-1]+=temp_var[(r*c*layer*filter)-1];

					}
				}
			}
		}

	}
	





}
