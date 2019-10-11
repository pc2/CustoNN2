# Performance Modeling Tool 

- In order to develop and launch our kernels in the most efficient way , we need to have a rough model of  how efficient our code is going to be.
- One way of doing this is to look at the source code and see the total number of operations in our file (ignoring overhead operations like loop index calulations etc).
- Doing this manually is very time consuming and boring. This tool is the first step towards automating  "Performance Modelling".
- This tool calculates the number of MAC ops in a file. 
- This tool is built upon Antlr4.
- Since OpenCL is just an extended version of C/C++ , we can make use of C grammar file available on the Internet to create lexers and parsers.
- We first have to analyse how our kernel code looks like. This means manually calculating number of MAC operations on a small kernel file.
- When we did this , we saw that all MACs happen inside the inner most loop of Conv kernels.
- So the simplest rule is to see how many times this inner most statement gets executed.
- There are multiple nested for loops in Conv kernels.
- So the rule to calculate MACs is to first grab hold of all the loop limits. If the for loops are nested , then we are supposed to multiply the loop limits. If the for loops are not nested , we add up the loop limits. We also have to consider the stride in these calulations.
- The tool has been fine tuned to execute according to the above rule.


## Antlr4 Basics : https://tomassetti.me/antlr-mega-tutorial/

## Steps
- We build lexer and parser for OpenCL file using the C.g4 grammar file.
- Antlr4 creates the lexer and parser for us.
- It generates these lexers and parsers in many languages. We have used Python2.7 for our tool.
- We make us of the CListener to walk the parse tree.
- We visit every node in the tree to check where we are in the tree
- If we are inside a function definition , we grab the name of the function and put it in a list.
- If we enter a for loop , we check for its nestedness using flags (which are updated as and when we enter and exit nodes). The implementation details are complicated. But suffice to say that we know the nestedness of loops.
- We grab loop limits and put them in a data strucutre to tell how many times a loop will be executed.
- Finally , we filter out Conv functions and display the operations - We need to multiply this by two if we consider MAC as two separate operations (the way it should be in HPC)
   
  

- The tool can be invoked in two modes - verbose or non verbose
- In verbose mode, we get more fine grained info about operations in a file
Command to invoke the tool :  

`python OpsCalculator.py -kernelfile <path to kernel file> [-v]`