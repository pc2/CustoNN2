// Example program
#include <iostream>
#include <string>
using namespace std;

float Performance_calculator(float fmax, float clock,int total_cycles, float time,int total_ops, float ops_per_cycle, int bytes, float ops_per_byte,float measured_runtime, float measured_cycles, float measured_ops_cycle, float measured_ops_sec )
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
    
    cout << "Enter total number of bytes: ";
    cin >> bytes;
    ops_per_byte = total_ops / bytes;
    cout << "Operations per byte are: " << ops_per_byte <<endl;
    
    cout << "Enter measured runtime: ";
    cin >> measured_runtime;
    measured_cycles = measured_runtime / clock;
    cout << "Measured cycles are: " << measured_cycles <<endl;
    
    cout << "Enter total number of operations: ";
    cin >> total_ops;
    measured_ops_cycle = total_ops / measured_cycles ;
    cout << "Operations per cycle are: " << measured_ops_cycle <<endl;
    
    cout << "Operations per second are: ";
    measured_ops_sec = total_ops / measured_runtime ;
    cout << "Operations per sec are: " << measured_ops_sec <<endl;
}

int main()
{	
	float clock=0.0 ;
	int total_cycles =0;
	float time = 0.0;
	int total_ops=0;
	float ops_per_cycle=0.0;
	float fmax=0.0;
	int bytes=0;
	float ops_per_byte = 0.0;
	float measured_runtime = 0.0;
	float measured_cycles = 0.0;
	float measured_ops_cycle = 0.0;
	float measured_ops_sec = 0.0;
	Performance_calculator(fmax, clock, total_cycles, time, total_ops, ops_per_cycle,bytes,ops_per_byte,measured_runtime,measured_cycles,measured_ops_cycle,measured_ops_sec);	
}

 
