import tvm
import numpy as np
import os

tgt="aocl_sw_emu"

fadd = tvm.module.load("myadd.so")
fadd_dev = tvm.module.load("myadd.aocx")
fadd.import_module(fadd_dev)

ctx = tvm.context(tgt, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype("float32"), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype("float32"), ctx)
c = tvm.nd.array(np.zeros(n, dtype="float32"), ctx)

fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
