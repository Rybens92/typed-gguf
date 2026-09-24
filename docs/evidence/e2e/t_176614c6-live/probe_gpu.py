"""Which GPU-facing libraries can this sandbox see? (host-side VRAM pressure needs a lever.)"""
import ctypes
import ctypes.util
import pathlib
import subprocess

for name in ("cuda", "nvidia-ml", "vulkan", "ggml"):
    print(f"find_library({name}) = {ctypes.util.find_library(name)}")
for path in sorted(pathlib.Path("/usr/lib64").glob("libcuda*")) + sorted(
        pathlib.Path("/usr/lib/x86_64-linux-gnu").glob("libcuda*")):
    print("lib:", path)
print("dev:", sorted(str(p) for p in pathlib.Path("/dev").glob("nvidia*")))
out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
print("nvidia-smi -L:", out.stdout.strip(), out.stderr.strip())
