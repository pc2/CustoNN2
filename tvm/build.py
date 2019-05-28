import tvm

tgt_host="llvm"
tgt="aocl_sw_emu"


n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = tvm.create_schedule(C.op)
px, x = s[C].split(C.op.axis[0], nparts=1)

s[C].bind(px, tvm.thread_axis("pipeline"))

fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")



fadd.save("myadd.o")
#fadd.imported_modules[0].save("myadd.aocx")

#tvm.contrib.cc.create_shared("myadd.so", ["myadd.o"])
