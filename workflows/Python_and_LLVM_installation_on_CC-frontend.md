### Installation steps to install Python3.6.7 for local user on CC-Frontend
- Download Python from [source](https://www.python.org/downloads/release/python-367/).
- Unzip or untar it in your local directory
- `cd $HOME`
- `mkdir .local` in your `$HOME` directory.
- `cd Python-3.6.7`
- `make clean`
- `./configure --prefix=/home/.local`
- `make`
- `make install`
- `PATH=$PATH:home/.local/bin/python3.6`



### Installation steps to install LLVM for local user on CC-Frontend
- Download LLVM from [source](http://releases.llvm.org/download.html)
- We have used LLVM 7.0.0 source code.
- `tar xf llvm-7.0.0.src.tar.xz`
- `mkdir mybuilddir`
- `cd mybuilddir`
- `cmake3 path/to/llvm/source/root` in our case its `llvm-7.0.0.src`
- `cmake3 --build .`
- `cmake3 --build . --target install`

[LLVM installation guide ](https://llvm.org/docs/CMake.html)