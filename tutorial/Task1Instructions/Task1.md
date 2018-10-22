# Task 1

The exercises are based on the Intel OpenCL FPGA training presented by El Camino based upon material by Intel. Adaptions are made to match our development platform and way of task distribution, but many steps use scans of the original instructions.

## Exercise 1

In this exercise you will practice writing and OpenCL host-side application including constructs to get the OpenCL platform and device; create an OpenCL context, command queue and buffers. We will use the C++ API in this lab.

### Step 1A, Preparation

- Connect to `fe-1.cc.pc2.uni-paderborn.de` with xrdp.
- Open a terminal, all following steps are performed in the same terminal, don't close it or you will have to go through most steps again.
- cd to your local copy of the repository.
- Fetch the exercise code templates and project files from the remote location with `git pull`.
- Within the repository change your path to Task1 with `cd tutorial/Task1`.
- Examine the file `opencl_init.sh` using a text editor (for example gedit by typing `gedit opencl_init.sh`).
	- Do you recognize parts of the script from the wiki documentation?
	- Quit gedit.
- Execute the script with `source opencl_init.sh`.
- Activate the eclipse environment with `scl enable rh-eclipse46 bash`.
- Launch eclipe with `eclipse &`.

### Step 1B, Working with Eclipse

- Set the workspace to `<path-to-your-repository>/tutorial/Task1`.
- If you are greeted by a welcome screen, close it by clicking the X in the tab that says welcome.
- You will see a project named **SimpleOpenCL** open in the **Project Explorer** pane on the left. You may need to refresh it with F5 or right-click on the project and selecting refresh.
- The **SimpleOpenCL** project is the project we're going to use to compile and run our OpenCL host program. You should notice that there are three source files **utility.cpp** and **utility.h** contain functions that perform useful tasks such as printing information about the OpenCL environment, **main.cpp* has a main() function that will contain all of our host side code, which is what we are going to modify in this exercise.
- Examine the project settings according to the following instructions. You may notice small differences to the screenshots. Do you have an idea of what they signify? ![Page 7.](Scan7.pdf)
<object data="Scan7.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan7.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan7.pdf">Download PDF</a>.</p>
    </embed>
</object>
![Page 8.](Scan8.pdf)
<object data="Scan8.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan8.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan8.pdf">Download PDF</a>.</p>
    </embed>
</object>
### Step 2, Write the host side code
*If you prefer to edit your code in another editor, it is fine. Simply navigate to the SimpleOpenCL/ subdirectory and open the files from there.*

- Open **main.cpp* from the **Project Explorer** pane in Eclipse by double-clicking on its name.
- Look for the comment "Exercise 1 Step 2.3 in **main.cpp**.
- Right below the comment, complete the line of code that will fill in the values for the vector **PlatformList**. Set the value returned  to the variable **err**. *Remember this lab file is designed to use the OpenCL C++ API as introduced on the tutorial slides. Since we statically linked to the Intel FPGA libraries, we know for sure there is only one platform, the statements afterwards assert that.*
- Look for the comment "Exercise 1 Step 2.5" in **main.cpp*.
- Write code that will fill in the values for the vector **DeviceList** below the comment "Exercise 1, Step 2.5". *Hint: You can use the device type CL_ DEVICE_ TYPE_ALL. Since we are going to execute this program in emulation mode for which we've set the number of devices to 1, there will only be 1 device.*
- Create a cl::Context object called **mycontext** below the comment "Exercise 1, Step 2.6". Pass the DeviceList into the constructor. *Everything we wrote up until this point is considered setup code using OpenCL platform layer APIs. This is code that only needs to be written once and can be used in may application scenarios. In your own code, you would likely store the setup code in a function so it can be easily reused. You will need a more flexible approach to handle different platform names and different numbers of devices though.*
- Create a command queue named *myqueue* below the comment "Exercise 1, Step 2.7". Use only the first (0th) device in the DeviceList.
- Below the comment "Exercise 1, Step 2.8", create three buffers named **Buffer_In**, **Buffer_In2**, and **Buffer_Out**. The first two buffers are READ_ONLY and Buffer_Out is WRITE_ONLY. The size of the buffer for all should be sizeof(cl_float)*vectorSize which is the total size of the array in bytes. *These are used to represent memory on the device and will be used as arguments to the kernel later.
- Below the comment "Exercise 1, Step 2.9", write two lines that would transfer the contents of **X** and **Y** into the buffers **Buffer_In** and **Buffer_In2** on the device respectively. Remember that in C++, an array's name is a pointer. Note: X, Y, and Z are arrays of floating point numbers. The function **fill_generated** randomly generates numbers between the values of **LO** and **HI** to fill the array.
![Page 11.](Scan11.pdf)
<object data="Scan11.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan11.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan11.pdf">Download PDF</a>.</p>
    </embed>
</object>
![Page 12.](Scan12.pdf)
<object data="Scan12.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan12.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan12.pdf">Download PDF</a>.</p>
    </embed>
</object>
![Page 13.](Scan13.pdf)
<object data="Scan13.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan13.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan13.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Exercise 2
- The displayed paths need to be adapted to `<path-to-your-repository>/tutorial/Task1`.
- In Step 1.8, you need to use a different call to aoc, namely `aoc -march=emulator -board=p385a_sch_ax115 SimpleKernel.cl`. Again: can you explain the change to the original description?
![Page 16.](Scan16.pdf)
<object data="Scan16.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan16.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan16.pdf">Download PDF</a>.</p>
    </embed>
</object>
![Page 17.](Scan17.pdf)
<object data="Scan17.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan17.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan17.pdf">Download PDF</a>.</p>
    </embed>
</object>
![Page 18.](Scan18.pdf)
<object data="Scan18.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan18.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan18.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Exercise 3
- The same changes apply as for Exercise 2.
- The displayed paths need to be adapted to `<path-to-your-repository>/tutorial/Task1`.
- In Step 1.8, you need to use a different call to aoc, namely `aoc -march=emulator -board=p385a_sch_ax115 SimpleKernel.cl`.
![Page 22.](Scan22.pdf)
<object data="Scan22.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan22.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan22.pdf">Download PDF</a>.</p>
    </embed>
</object>
![Page 23.](Scan23.pdf)
<object data="Scan23.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan23.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan23.pdf">Download PDF</a>.</p>
    </embed>
</object>
![Page 24.](Scan24.pdf)
<object data="Scan24.pdf" type="application/pdf" width="700px" height="1000px">
    <embed src="Scan24.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="Scan24.pdf">Download PDF</a>.</p>
    </embed>
</object>