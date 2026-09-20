#!/usr/bin/env python3
"""Audit helper (t_78f5ea7a): does dlopen dedupe two bundles by SONAME?

Loads libllama.so from the CPU bundle and from the Vulkan bundle in one process and prints
whether the dynamic loader handed back the same object (same handle), which would mean a
'vulkan' session can silently run on the CPU bundle's code.
"""
import ctypes
import pathlib

CPU = pathlib.Path("/tmp/e2-audit/runtime/b11026-linux-x64-cpu")
VULKAN = pathlib.Path("/var/home/rybens/.local/share/ggufone/runtime/b11026-linux-x64-vulkan")

cpu_llama = ctypes.CDLL(str(CPU / "libllama.so"), mode=ctypes.RTLD_GLOBAL)
vulkan_llama = ctypes.CDLL(str(VULKAN / "libllama.so"), mode=ctypes.RTLD_GLOBAL)
print("cpu    handle:", cpu_llama._handle, "->", CPU / "libllama.so")
print("vulkan handle:", vulkan_llama._handle, "->", VULKAN / "libllama.so")
print("same object (dedupe by SONAME):", cpu_llama._handle == vulkan_llama._handle)

cpu_ggml = ctypes.CDLL(str(CPU / "libggml.so"), mode=ctypes.RTLD_GLOBAL)
vulkan_ggml = ctypes.CDLL(str(VULKAN / "libggml.so"), mode=ctypes.RTLD_GLOBAL)
print("ggml same object (dedupe by SONAME):", cpu_ggml._handle == vulkan_ggml._handle)
