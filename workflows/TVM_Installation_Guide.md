# TVM Installation guide on CC Frontend

Step 1. Clone tvm repository from git



    git clone --recursive https://github.com/dmlc/tvm


Step 2. Create build direcotry and copy cmake/config.cmake file

---

mkdir build
cp cmake/config.cmake build

---

Step 3. Change set(USE_OPENCL OFF) to set(USE_OPENCL ON)
Change set(USE_AOCL OFF) to set(USE_AOCL ON)

Step 4. 

---

cd build
cmake ..
make -j4

---

Step 5. 

---
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}

cd python; python setup.py install --user; cd ..
cd topi/python; python setup.py install --user; cd ../..
cd nnvm/python; python setup.py install --user; cd ../..

---