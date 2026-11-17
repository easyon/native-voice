import os
import time

import torch

print("torch", torch.__version__)
print("num_threads", torch.get_num_threads())
print("interop", torch.get_num_interop_threads())
print("mkldnn", torch.backends.mkldnn.is_available())
print("cpu count", os.cpu_count())
x = torch.randn(1024, 1024)
y = torch.randn(1024, 1024)
for _ in range(3):
    z = x @ y
t0 = time.time()
n = 50
for _ in range(n):
    z = x @ y
dt = time.time() - t0
gflops = n * 2 * 1024**3 / dt / 1e9
print(f"matmul 1024^3 x{n}: {dt:.2f}s -> {gflops:.1f} GFLOPS")
for nt in (4, 8, 12, 16):
    torch.set_num_threads(nt)
    t0 = time.time()
    for _ in range(n):
        z = x @ y
    dt = time.time() - t0
    print(f"threads={nt}: {n*2*1024**3/dt/1e9:.1f} GFLOPS")
print("final threads", torch.get_num_threads())
