type_flag = input("Enter type of kernel (conv or pool)")
if(type_flag == "conv"):
    dim = input("enter size of conv")
    out_dims = []
    temp_dims = input("enter output dimensions separated by space")
    out_dims = temp_dims.split()
    in_dims = []
    temp_dims2 = input("enter input dimensions separated by space")
    in_dims = temp_dims2.split()
    time_in_ms = input("Enter execution time in ms")
    freq = input("Enter frequency in MHz")
    num_ops = 2 * int(dim) * int(dim) * int(out_dims[0]) * int(out_dims[0]) * int(in_dims[2]) * int(out_dims[2])
    num_ops = num_ops + (int(dim)*int(out_dims[0])*int(out_dims[0])*int(in_dims[2])*int(out_dims[2]))
    num_ops = num_ops + (2*int(out_dims[0])*int(out_dims[0])*int(in_dims[2])*int(out_dims[2]))
    num_ops = num_ops + (int(out_dims[0])*int(out_dims[0])*int(out_dims[2]))
    print("Total operations: "+str(num_ops/1000000)+" million")
    total_bytes_read = 4 * int(out_dims[2]) + 4 * int(dim) * int(dim) * int(in_dims[2]) * int(out_dims[2])+ 4 * int(in_dims[0])*int(in_dims[0])*int(in_dims[2]) * int(out_dims[2])
    total_bytes_written = 4 * int(out_dims[0]) * int(out_dims[0]) * int(out_dims[2])
    print("Global memory R: "+str(total_bytes_read/100000)+" MB")
    print("Global memory W: " + str(total_bytes_written / 100000)+" MB")
    num_cycles = float(freq) * pow(10,6) * float(time_in_ms) * pow(10,-3)
    print("Execution time: "+str(time_in_ms)+" ms")
    print("ops per second (G): "+str((num_ops/(float(time_in_ms)*pow(10,-3)))*pow(10,-9)))
    print("ops per cycle: "+str(num_ops/num_cycles))
    print("ops per byte: "+str(num_ops/(total_bytes_read+total_bytes_written)))
    print("Global memory per second (GB): "+ str( ( ( total_bytes_read + total_bytes_written ) / ( float(time_in_ms) * pow(10,-3)) ) * pow(10,-9) ) )
    print("Global memory per cycle: "+str((total_bytes_written+total_bytes_read)/num_cycles))
elif(type_flag=="pool"):
    temp_dims = input("enter output dimensions separated by space")
    out_dims = temp_dims.split()
    in_dims = []
    temp_dims2 = input("enter input dimensions separated by space")
    in_dims = temp_dims2.split()
    time_in_ms = input("Enter execution time in ms")
    freq = input("Enter frequency in MHz")
    total_bytes_read = 4*int(in_dims[0])*int(in_dims[0])*int(out_dims[2])
    total_bytes_written = 4 * int(out_dims[0])*int(out_dims[0])*int(out_dims[2])
    print("Global memory R: "+str(total_bytes_read))
    print("Global memory W: " + str(total_bytes_written))
    num_cycles = float(freq) * pow(10, 6) * float(time_in_ms) * pow(10, -3)
    print("Execution time: " + str(time_in_ms) + " ms")
    print("Global memory per second: " + str((total_bytes_read + total_bytes_written) / (float(time_in_ms) * pow(10, -3))))
    print("Global memory per cycle: " + str((total_bytes_written + total_bytes_read) / num_cycles))