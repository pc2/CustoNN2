#pragma OPENCL EXTENSION cl_intel_channels : enable

channel short16 serial_in0    __attribute__((depth(0))) __attribute__((io("kernel_input_ch0")));    //Channel 0 Rx
channel short16 serial_out0   __attribute__((depth(0))) __attribute__((io("kernel_output_ch0")));  //Channel 0 Tx
channel short16 serial_in1    __attribute__((depth(0))) __attribute__((io("kernel_input_ch1")));   //Channel 1 Rx
channel short16 serial_out1   __attribute__((depth(0))) __attribute__((io("kernel_output_ch1"))); //Channel 1 Tx

channel short16 serial_in2    __attribute__((depth(0))) __attribute__((io("kernel_input_ch2")));    //Channel 2 Rx
channel short16 serial_out2   __attribute__((depth(0))) __attribute__((io("kernel_output_ch2")));  //Channel 2 Tx
channel short16 serial_in3    __attribute__((depth(0))) __attribute__((io("kernel_input_ch3")));   //Channel 3 Rx
channel short16 serial_out3   __attribute__((depth(0))) __attribute__((io("kernel_output_ch3"))); //Channel 3 Tx

// Not supporting emulation at this point.
#ifdef EMULATION
#endif


/*
    Using short16 for testing 256 bit interface.
*/
__kernel void do_work(__global unsigned int *restrict latency)
{
    short16 reg;
    reg.s0 = 0xbeef;
    unsigned int clock_count = 0;
    bool have_data = false;
    bool sent_data = false;
    // Loop will stop once data has been received
    while (!have_data)
    {
        if (!sent_data)
        {
            write_channel_intel(serial_out0, reg);
            sent_data = true;
        }
        clock_count++;
        // have_data will be true once serial channel completes transfer
        read_channel_nb_intel(serial_in1, &have_data);
    }
    latency[0] = clock_count;
}
