"""HuggingFace resolve/search/download with resume + SHA-256 verify

Milestone: E1a.

SPEC 2.7: download exactly the selected quant file, verify against the tree
API lfs.oid, publish atomically via os.replace.
"""
