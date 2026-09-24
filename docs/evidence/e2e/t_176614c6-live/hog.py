"""Hold N MiB of device memory for N seconds (a stand-in for "the desktop took the card").

Not part of the product: a *test harness* that creates the (b) state on demand — a host whose
weights are resident but whose per-request context no longer fits. Uses libcuda directly (no
torch), so it starts in milliseconds and holds nothing but the allocation.
"""
import ctypes
import sys
import time

mib = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0

cuda = ctypes.CDLL("libcuda.so.1")
rc = cuda.cuInit(0)
if rc != 0:
    raise SystemExit(f"cuInit failed: {rc}")
device = ctypes.c_int(0)
rc = cuda.cuDeviceGet(ctypes.byref(device), 0)
if rc != 0:
    raise SystemExit(f"cuDeviceGet failed: {rc}")
size = mib * 1024 * 1024
ptr = ctypes.c_void_p(0)
rc = cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(size))
if rc != 0:
    raise SystemExit(f"cuMemAlloc({mib} MiB) failed: rc={rc} (free device memory is short)")
print(f"hog: holding {mib} MiB at {ptr.value:#x} for {seconds}s", flush=True)
time.sleep(seconds)
cuda.cuMemFree_v2(ptr)
print("hog: released", flush=True)
