"""The process's own teardown policy: who decides how a typed-gguf process ends (card t_97f1bc93).

`runtime.isolated` answered one half of this question for the probes: a C library must never be
able to kill the *caller*, so every probe runs in a disposable child — "it dies alone". This module
is the other half, for the processes that *are* the answer: the command's exit status and its
streams belong to the command, and a third-party destructor must not rewrite either.

Measured on the operator's box (`typed-gguf bench --backend vulkan`, one bundle, a 4B model, a
device that is nearly full): the whole report reaches stdout, then the process dies with
**SIGSEGV** —
`exit 139`, so the isolation layer withholds the row ("the exit code and the report contradict each
other"). Captured backtrace:

```
######## SEGV_BT: signal 11 (Segmentation fault) fault_address=0x18 ########
/usr/lib64/libnvidia-glvkspirv.so.615.71.09(+0x363b2)
/usr/lib64/libnvidia-eglcore.so.615.71.09(+0xcf48df)
/usr/lib64/libnvidia-eglcore.so.615.71.09(+0xcf6db5)
/usr/lib64/libnvidia-eglcore.so.615.71.09(+0x97edea)
/lib/x86_64-linux-gnu/libc.so.6(+0x92b7b)          <- __run_exit_handlers
/lib/x86_64-linux-gnu/libc.so.6(+0x1107f8)         <- __libc_start_main
```

No ggml, no llama.cpp, no typed-gguf frame is in it: the fault is the NVIDIA ICD's **own exit
handler**, i.e. exactly the class of code `runtime.isolated` already refuses to trust. It is
intermittent (roughly a third of the runs on that box — the count is in
`.e2e/t_97f1bc93-vulkan-teardown/`), which is why the row's contradiction is the only description
of it a user ever gets.

So: a process that has dlopened a bundle ends **itself**, with the code the command produced and
both streams flushed, before any exit handler runs (`end_process`). Nothing that could fail
*earlier* is hidden by it — an exception, a signal during the run, a withheld row all still reach
the shell first; this only takes away a third party's vote on the exit status. A process that never
loaded a bundle has nothing to escape from and keeps the interpreter's normal shutdown.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import NoReturn

from typed_gguf.runtime import ctypes_binding


def engine_loaded() -> bool:
    """True when this process has dlopened a local llama.cpp bundle (so its destructors exist)."""
    return bool(ctypes_binding.loaded_runtimes())


def end_process(code: int) -> NoReturn:
    """End the process with `code`, without running exit handlers.

    `os._exit` *is* the remedy: it skips `atexit` callbacks and every destructor registered by the
    libraries this process dlopened — the level at which the measured SIGSEGV lives. Nothing of the
    command's own answer is skipped with them: the report was written (and its file closed) during
    the command, and both streams are flushed here before the process ends, so a piped stdout keeps
    the bytes the reader came for (`python -m typed_gguf bench … --json | jq .` must not read a
    truncated document because the answer arrived in a block-buffered pipe).
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):   # a closed or None stream must not change the code
            stream.flush()
    os._exit(int(code))
