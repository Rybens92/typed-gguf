"""What does the pinned SDK's client offer for timeouts / retries?"""
from __future__ import annotations

import inspect

import typesafe_sdk
from typesafe_sdk import TypeSafeClient

print("typesafe-sdk", typesafe_sdk.__version__)
print("TypeSafeClient.__init__", inspect.signature(TypeSafeClient.__init__))
print()
print(inspect.getsource(TypeSafeClient)[:2000])
