TVM uses `.egg` files for distributing bundles. It is similar to '.jar' . TVM has its codebase in C++ with API support mainly with python. 
Whenever we provide a frozen `.protobuf` file to TVM, it is parsed by tensorflow for generating the graph definition from it and based on its last node which we provide, output shape is added to it. 

We have used the NNVM compiler for the generation of our kernels.
`nnvm.frontend` has support for different machine learning framework (TensorFlow, ONNX) and based on the graph definition compiler calls a helper class `graphproto` for converting into `nnvm.symbol` and `parameters` dict of pre-trained weights. This parameters are in `tvm.nd.array` format. `nnvm/frontend/tensorflow.py` is accessed during this process. 

For generating `nnvm.symbol` a C++ api call is registered in `lib/nnvm/graph.py` :
```
    @property
    def symbol(self):
        shandle = SymbolHandle()
        check_call(_LIB.NNGraphGetSymbol(self.handle, ctypes.byref(shandle)))
        return Symbol(shandle)
```

Which in turns calls `nnvm/src/c_api/c_api_symbolic.cc`

The symbol and parameters generated are given to `nnvm_compiler` along with its optimization level-

0 - SimplifyInference

1 - OpFusion

2 - PrecomputePrune

3 - FoldScaleAxis

To start with the compilation process `nnvm/compiler/build_module.py` is initiated and Class `BuildConfig` is called during this process. TVM register its API calls under `/src/api/api_lang.cc` along with a `PackedFunc` . A `PackedFunc` makes the API calls typed erased and wrap around TVM type class around its input and outputs.

```
TVM_REGISTER_API("_Placeholder")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    *ret = placeholder(args[0],
                       args[1],
                       args[2]);
  });

```
`TVMArgs` and `TVMRetValue` are packed functions for input and output respectively.

After registering a API call to `BuildConfig` in c++ codebase `src/codegen/build_module.cc` a `NodeBase` is created . `NodeBase` is a base class where are the datatypes are made subset of it eg- `str`,`float`,etc. `NodeBase` lies under `python/tvm/_ffi/`.

After this `BuildStmt` is called in `build_module.cc` which first calls the optimization level `nnvm\src\compiler\fold_scale_axis.cc` and on return of the api calls for `@tvm.register_func("nnvm.compiler.lower")` is registered which transforms high level IR and nested loop strucutres to low level IR. 

Once lowering is completed the `nnvm.comipler.build` checks for the `target_host` and `_init_api("tvm.codegen")` api call is made with lowered function and target host.  The `Build()` is mapped to `src/codegen/codegen.cc`

```
runtime::Module Build(const Array<LoweredFunc>& funcs,
                      const std::string& target) {
  std::string mode = target;
  size_t pos = mode.find(' ');
  if (pos != std::string::npos) {
    mode = mode.substr(0, pos);
  }
  std::string build_f_name = "codegen.build_" + mode;
  // the build function.
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  CHECK(bf != nullptr)
      << "Target " << target << " is not enabled";
  runtime::Module m = (*bf)(funcs, target);
  return m;
}
```

In our scenario, `codegen.build_aocl` is mapped which calls `src/codegen/codegen_aocl.cc`

```
TVM_REGISTER_API("codegen.build_aocl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildAOCL(args[0], args[1], false);
  });
```

which in returns calls `BuildAOCL` in the same file for the generation of `aocl.cl` kernel file.


