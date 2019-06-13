import tvm

tgt_host="llvm"
tgt="aocl_sw_emu"

n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] / B[i], name="C")
D = tvm.compute(C.shape, lambda i: C[i] + A[i], name="D")

s = tvm.create_schedule(C.op)
v = tvm.create_schedule(D.op)
px, x = s[C].split(C.op.axis[0], nparts=1)
qx, y = s[D].split(D.op.axis[0], nparts=1)

s[C].bind(px, tvm.thread_axis("pipeline"))
v[D].bind(qx, tvm.thread_axis("pipeline"))

fadd = tvm.build(s, [A, B, C, D], tgt, target_host=tgt_host, name="myadd")

#faddsec = tvm.build(s, [A, B, C, D], tgt, target_host=tgt_host, name="secondadd")

#fadd.save("myadd.o")
#fadd.imported_modules[0].save("myadd.aocx")

#tvm.contrib.cc.create_shared("myadd.so", ["myadd.o"])


