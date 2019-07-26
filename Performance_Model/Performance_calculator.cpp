#include<iostream>
#include<math.h>
using namespace std;

float Performance_calculator(float fmax, float clock,int total_cycles, float time,int total_ops, float ops_per_cycle )
{
    cout << "Enter Fmax value: ";
    cin >> fmax;
    clock = 1 / fmax;
    cout << "One clock cycle in sec: " << clock <<endl;
    
    
	cout << "Enter total number of cycles: ";
    cin >> total_cycles;
	time = clock * total_cycles;
	cout << "Total execution time in seconds: " << time <<endl;
	
	
	cout << "Enter total number of operations: ";
    cin >> total_ops;
    ops_per_cycle = total_ops / total_cycles;
    cout << "Operations per cycle are: " << ops_per_cycle <<endl;
}

int main()
{	
	float clock=0.0 ;
	int total_cycles =0;
	float time = 0.0;
	int total_ops=0;
	float ops_per_cycle=0.0;
	float fmax=0.0;
	Performance_calculator(fmax, clock, total_cycles, time, total_ops, ops_per_cycle);	
}