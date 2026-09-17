"""GGUF header/metadata reader (SPEC 2.7, A-E1a-10).

Synthetic headers cover v2/v3, every scalar type, arrays, nested arrays and truncation;
real pinned files are exercised behind @pytest.mark.model.
"""
from __future__ import annotations

import hashlib
import pathlib
import struct

import pytest

from ggufone.errors import GgufoneError
from ggufone.registry.gguf import (
    FTYPE_NAMES,
    GGUF_MAGIC,
    parse_gguf_metadata,
    quant_label,
    sha256_file,
)

T_STRING, T_ARRAY = 8, 9
WORKDIR: pathlib.Path = pathlib.Path(".")


@pytest.fixture(autouse=True)
def _workdir(tmp_path: pathlib.Path) -> None:
    global WORKDIR
    WORKDIR = tmp_path


def gstr(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv(key: str, type_id: int, payload: bytes) -> bytes:
    return gstr(key) + struct.pack("<I", type_id) + payload


def array(elem_type: int, items: list[bytes]) -> bytes:
    return struct.pack("<IQ", elem_type, len(items)) + b"".join(items)


def build(*, version: int = 3, n_tensors: int = 2, kvs: list[bytes] = (), magic: bytes = GGUF_MAGIC,
          n_kv: int | None = None, trailer: bytes = b"") -> bytes:
    count = len(kvs) if n_kv is None else n_kv
    return (magic + struct.pack("<I", version) + struct.pack("<Q", n_tensors)
            + struct.pack("<Q", count) + b"".join(kvs) + trailer)


def write(blob: bytes, name: str = "synthetic.gguf") -> pathlib.Path:
    path = WORKDIR / name
    path.write_bytes(blob)
    return path


# ------------------------------------------------------------------ basics
def test_reads_flat_header_v3() -> None:
    blob = build(kvs=[
        kv("general.architecture", T_STRING, gstr("spark2_5")),
        kv("general.file_type", 4, struct.pack("<I", 7)),
        kv("spark2_5.block_count", 4, struct.pack("<I", 36)),
    ])
    got = parse_gguf_metadata(write(blob))
    assert got["version"] == 3
    assert got["n_tensors"] == 2
    assert got["n_kv"] == 3
    assert got["kv"]["general.architecture"] == "spark2_5"
    assert got["kv"]["general.file_type"] == 7
    assert got["kv"]["spark2_5.block_count"] == 36


def test_reads_version_2_header() -> None:
    got = parse_gguf_metadata(write(build(version=2, kvs=[kv("a", T_STRING, gstr("b"))])))
    assert got["version"] == 2 and got["kv"] == {"a": "b"}


def test_reads_every_scalar_type() -> None:
    kvs = [
        kv("u8", 0, struct.pack("<B", 255)),
        kv("i8", 1, struct.pack("<b", -5)),
        kv("u16", 2, struct.pack("<H", 65535)),
        kv("i16", 3, struct.pack("<h", -32768)),
        kv("u32", 4, struct.pack("<I", 4_000_000_000)),
        kv("i32", 5, struct.pack("<i", -2_000_000_000)),
        kv("f32", 6, struct.pack("<f", 1.5)),
        kv("bool", 7, struct.pack("<?", True)),
        kv("u64", 10, struct.pack("<Q", 2 ** 63)),
        kv("i64", 11, struct.pack("<q", -(2 ** 62))),
        kv("f64", 12, struct.pack("<d", 0.5)),
    ]
    kv_map = parse_gguf_metadata(write(build(kvs=kvs)))["kv"]
    assert kv_map["u8"] == 255 and kv_map["i8"] == -5
    assert kv_map["u16"] == 65535 and kv_map["i16"] == -32768
    assert kv_map["u32"] == 4_000_000_000 and kv_map["i32"] == -2_000_000_000
    assert kv_map["f32"] == 1.5 and kv_map["bool"] is True
    assert kv_map["u64"] == 2 ** 63 and kv_map["i64"] == -(2 ** 62) and kv_map["f64"] == 0.5


def test_reads_array_of_strings_including_empty_and_unicode() -> None:
    blob = build(kvs=[kv("tokens", T_ARRAY, array(T_STRING, [gstr("a"), gstr(""), gstr("Δ")]))])
    assert parse_gguf_metadata(write(blob))["kv"]["tokens"] == ["a", "", "Δ"]


def test_reads_nested_arrays() -> None:
    inner = array(T_STRING, [gstr("x"), gstr("y")])
    blob = build(kvs=[kv("grid", T_ARRAY, array(T_ARRAY, [inner, array(T_STRING, [])]))])
    assert parse_gguf_metadata(write(blob))["kv"]["grid"] == [["x", "y"], []]


def test_stops_after_n_kv_and_never_reads_tensor_data() -> None:
    blob = build(kvs=[kv("a", T_STRING, gstr("b"))], n_kv=1, trailer=b"\x00" * 4096)
    assert parse_gguf_metadata(write(blob))["kv"] == {"a": "b"}


# ------------------------------------------------------------------ corruption
def test_bad_magic_is_corrupt() -> None:
    with pytest.raises(GgufoneError) as exc:
        parse_gguf_metadata(write(build(magic=b"GGOX")))
    assert exc.value.code == "E_GGUF_CORRUPT"


def test_truncated_header_is_corrupt() -> None:
    blob = build(kvs=[kv("general.architecture", T_STRING, gstr("spark2_5"))])
    with pytest.raises(GgufoneError) as exc:
        parse_gguf_metadata(write(blob[: len(blob) - 4]))
    assert exc.value.code == "E_GGUF_CORRUPT"


def test_truncated_string_payload_is_corrupt() -> None:
    blob = build(kvs=[kv("a", T_STRING, gstr("hello"))])
    with pytest.raises(GgufoneError) as exc:
        parse_gguf_metadata(write(blob[: len(blob) - 3]))
    assert exc.value.code == "E_GGUF_CORRUPT"


def test_unknown_value_type_is_corrupt() -> None:
    with pytest.raises(GgufoneError) as exc:
        parse_gguf_metadata(write(build(kvs=[kv("a", 99, b"\x00\x00\x00\x00")])))
    assert exc.value.code == "E_GGUF_CORRUPT"


def test_unsupported_version_is_corrupt() -> None:
    with pytest.raises(GgufoneError) as exc:
        parse_gguf_metadata(write(build(version=99, kvs=[kv("a", T_STRING, gstr("b"))])))
    assert exc.value.code == "E_GGUF_CORRUPT"
    assert "99" in str(exc.value)


def test_missing_file_is_corrupt_with_path_in_message(tmp_path: pathlib.Path) -> None:
    with pytest.raises(GgufoneError) as exc:
        parse_gguf_metadata(tmp_path / "absent.gguf")
    assert exc.value.code == "E_GGUF_CORRUPT"
    assert "absent.gguf" in str(exc.value)


def test_array_payload_truncated_is_corrupt() -> None:
    blob = build(kvs=[kv("t", T_ARRAY, array(T_STRING, [gstr("aa"), gstr("bb")]))])
    with pytest.raises(GgufoneError) as exc:
        parse_gguf_metadata(write(blob[: len(blob) - 2]))
    assert exc.value.code == "E_GGUF_CORRUPT"


def test_empty_file_is_corrupt() -> None:
    with pytest.raises(GgufoneError) as exc:
        parse_gguf_metadata(write(b""))
    assert exc.value.code == "E_GGUF_CORRUPT"


def test_absurd_n_kv_is_rejected_before_allocating() -> None:
    blob = GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2 ** 60)
    with pytest.raises(GgufoneError) as exc:
        parse_gguf_metadata(write(blob))
    assert exc.value.code == "E_GGUF_CORRUPT"


def test_reads_from_file_object(tmp_path: pathlib.Path) -> None:
    path = write(build(kvs=[kv("a", T_STRING, gstr("b"))]))
    with open(path, "rb") as fh:
        assert parse_gguf_metadata(fh)["kv"] == {"a": "b"}


# ------------------------------------------------------------------ ftype map
def test_ftype_table_pins_the_two_executed_values() -> None:
    assert FTYPE_NAMES[7] == "MOSTLY_Q8_0"
    assert FTYPE_NAMES[15] == "MOSTLY_Q4_K_M"
    assert quant_label(7) == "Q8_0"
    assert quant_label(15) == "Q4_K_M"
    assert quant_label(1) == "F16"
    assert quant_label(0) == "F32"
    assert quant_label(2) == "Q4_0"
    assert quant_label(1024) == "GUESSED"
    assert quant_label(12345) == "FTYPE_12345"


def test_ftype_table_covers_the_pinned_enum_range() -> None:
    for ftype in (0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30, 32, 36, 37, 38,
                  39, 40, 41):
        assert ftype in FTYPE_NAMES


# ------------------------------------------------------------------ sha256
def test_sha256_file_matches_hashlib(tmp_path: pathlib.Path) -> None:
    payload = b"ggufone" * 100_000
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
    assert sha256_file(path, chunk=7) == hashlib.sha256(payload).hexdigest()


# ------------------------------------------------------------------ real files
HOME = pathlib.Path.home()
PINNED_SPARK = HOME / ".hermes" / "models" / "Spark-X2.5-4B-Q8_0.gguf"
PINNED_QWEN = HOME / ".cache" / "llama.cpp" / "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"


@pytest.mark.model
@pytest.mark.skipif(not PINNED_SPARK.exists(), reason="pinned Spark GGUF not present")
def test_pinned_spark_header() -> None:
    kv = parse_gguf_metadata(PINNED_SPARK)["kv"]
    assert kv["general.architecture"] == "spark2_5"
    assert kv["general.file_type"] == 7
    assert kv["spark2_5.block_count"] == 36
    assert kv["spark2_5.attention.head_count"] == 16
    assert kv["spark2_5.attention.head_count_kv"] == 4
    assert kv["spark2_5.attention.key_length"] == 256
    assert kv["spark2_5.attention.value_length"] == 256
    assert kv["spark2_5.context_length"] == 1_048_576
    assert kv["spark2_5.embedding_length"] == 2560
    assert quant_label(kv["general.file_type"]) == "Q8_0"


@pytest.mark.model
@pytest.mark.skipif(not PINNED_QWEN.exists(), reason="pinned Qwen3.5 GGUF not present")
def test_pinned_qwen_header() -> None:
    kv = parse_gguf_metadata(PINNED_QWEN)["kv"]
    assert kv["general.architecture"] == "qwen35"
    assert kv["qwen35.block_count"] == 24
    assert quant_label(kv["general.file_type"]) == "Q4_K_M"
