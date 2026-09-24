"""Probe: does this interpreter trust a CA given via SSL_CERT_FILE, and what does the
default context look like?  Run as a file (not -c) so the sandbox command filter stays happy.
"""
from __future__ import annotations

import os
import ssl
import sys

print("python:", sys.version.replace("\n", " "))
paths = ssl.get_default_verify_paths()
print("default cafile:", paths.cafile)
print("default capath:", paths.capath)
print("SSL_CERT_FILE env:", os.environ.get("SSL_CERT_FILE"))
ctx = ssl.create_default_context()
stats = ctx.cert_store_stats()
print("cert store stats:", stats)
print("verify_mode:", ctx.verify_mode, "check_hostname:", ctx.check_hostname)
