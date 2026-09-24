"""Print the `413` reason phrase the *interpreter* owns — the fact behind card t_b5872762.

`typed_gguf.api.http` answers the body cap with `Handler.send_response(413)`, and
`BaseHTTPRequestHandler` takes the phrase from `http.HTTPStatus`. Run this with each interpreter of
the CI matrix to see the text differ:

    python probe_413_phrase.py
"""
import http
import http.server
import sys

status = http.HTTPStatus(413)
print(f"CPython {sys.version.split()[0]}")
print(f"  http.HTTPStatus(413).phrase  = {status.phrase!r}")
print(f"  handlers' own responses[413] = {http.server.BaseHTTPRequestHandler.responses[413][0]!r}")
