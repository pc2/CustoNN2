# Task 6

## Setup

- Create a new feature branch `task6_<yourimtname>`
- In that branch, create a copy of the Task6 folder named `Task6Solution<YourIMTname>`.
- Start Eclipse with a workspace at `Task6Solution<YourIMTname>` and import the SimpleOpenCL project to it.

## Introduce channels

- In the `SimpleKernel.cl` file, enable the Intel channel extension and create a channel of type `float` and call it `chan_in2`.
- In the `SimpleKernel.cl` file, create a second kernel called `InputKernel`.
- Use `in2` and `N` as input arguments for the new kernel.
- In a loop, read the N values from `in2` and write them to channel `chan_in2`.
- In `SimpleKernel`, remove in2 from the argument list. Replace the access to `in2` with a read from channel `chan_in2`.
- Adapt the host code to create and launch both kernels with the right arguments. Attention: how many command queues do you have to use?
- Compile kernel and host code and test for correctness.

## Wider Channel

- Copy your kernel file to `SimpleKernelN8.cl`.
- In this kernel, we want to process 8 elements per cycle instead of a single one.
- Come up with a solution how to achieve that and implement it.
- Compile kernel and host code and test for correctness, make sure to use the new kernel.
- Create a report and ensure that the initiation interval stays at 1.

## Submission

- Commit your working solution excluding all generated files.
- Push your feature branch to the remote repository.
- Create a merge request.