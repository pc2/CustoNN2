# Performance Modeling Tool 

- In order to develop and launch our kernels in the most efficient way , we need to have a rough model of  how efficient our code is going to be.
- One way of doing this is to look at the source code and see the total number of operations in our file (ignoring overhead operations like loop index calulations etc).
- Doing this manually is very time consuming and boring. This tool is the first step towards automating  "Performance Modelling".
- This tool calculates the number of MAC ops in a file. Multiplication and Addition is considered as one operation here.
- This tool is built upon Antlr4.
- Since OpenCL is just an extended version of C/C++ , we can make use of C grammar file available on the Internet to create lexers and parsers.


## Antlr4 Basics : https://tomassetti.me/antlr-mega-tutorial/

## Steps
- We build lexer and parser for OpenCL file using the C.g4 grammar file.
- Antlr4 creates the lexer and parser for us.
- It generates these lexers and parsers in many languages. We have used Python2.7 for our tool.
- We make us of the CListner to be able to walk the parse tree.


