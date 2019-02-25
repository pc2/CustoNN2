**Tensor virtual machine**
>Today we have a lot of different deep learning frameworks such as TensorFlow, Cafee2, MXNet and PyTorch. Mapping this frameworks to different devices is complicated by the diversity of hardware characteristics,
including embedded CPUs, GPUs, FPGAs, and ASICs.

>TVM is a compiler that takes deep learning frameworks and generates low-level optimized code for a diverse set of hardware back-ends.

>TVM was developed lie a research project at the Paul G. Allen School of Computer Science & Engineering, University of Washington.

>TVM is a open-source product.

>TVM made a several contributions like a computational graph rewriter; tensor expression language and new features for GPU accelerators.

>The system first takes as input a model from an existing framework and transforms it into a computational graph representation. It then performs high-level dataflow rewriting to generate an optimized graph. The operator-level optimization module must generate efficient code for each fused operator in this graph. Operators are specified in a declarative tensor expression language; execution details are unspecified. TVM identifies a collection of possible code optimizations for a given hardware target’s operators.

**Optimizing computational graphs**
>* operator fusion: Combines multiple operators into single kernel without saving intermediate results. 
>* constant-folding: Pre computing graph parts that can be determined statically thereby saving execution costs. 
>* static memory planning pass: Pre allocates memory to hold each intermediate tensor. 
>* data layout transformations: Converts a computational graph into one that can use better internal layouts for execution on target hardware. 

>By extending the TVM stack with a customizable, and open source deep learning hardware accelerator design, we are exposing a transparent end-to-end deep learning stack from the high-level deep learning framework, down to the actual hardware design and implementation. 

>So here we have VTA(versatile tensor accelerator), which is a complete deep learning system with TVM.

**VTA stack**

>1. NNVM: Acts as graph optimizer
>2. TVM IR: Provides portability across divergent hardware backends
>3. VTA JIT runtime: Complies VTA instruction stream and microkernels.

**Members**
>Alina Egorova </br>
>Arathy Ajaya Kumar
