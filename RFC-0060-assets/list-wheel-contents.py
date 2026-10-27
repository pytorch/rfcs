#!/usr/bin/env python3
"""List the contents of PyTorch wheels and libtorch zips without downloading them.

Reads each archive's zip central directory over HTTP range requests (a few
hundred KB per file instead of gigabytes) and writes one tab-separated
"<uncompressed size> TAB <path>" line per entry. This is how the per-directory
table in RFC-0060 was produced; rerun it for a later release to refresh the
numbers.

Usage:
    list-wheel-contents.py <version> [<outdir>]

Example:
    list-wheel-contents.py 2.14.0 ./listings
"""

import html
import re
import struct
import sys
import urllib.request
from pathlib import Path

BASE = "https://download.pytorch.org"
PYTHON_TAG = "cp312"


def get(url: str, headers: dict[str, str] | None = None) -> tuple[bytes, object]:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), r.headers


def index_links(index: str) -> list[str]:
    body, _ = get(index)
    out = []
    for href in re.findall(r'href="([^"]+)"', body.decode()):
        href = html.unescape(href).split("#")[0]
        if href.startswith("http"):
            out.append(href)
        elif href.startswith("/"):
            out.append(BASE + href)
        else:
            out.append(index.rstrip("/") + "/" + href)
    return out


def central_directory(url: str) -> tuple[int, list[tuple[str, int]]]:
    """Return (archive size, [(path, uncompressed size), ...]) for a remote zip."""
    _, h = get(url, {"Range": "bytes=0-0"})
    size = int(h["Content-Range"].split("/")[1])
    tail, _ = get(url, {"Range": f"bytes={max(0, size - 70000)}-{size - 1}"})
    p = tail.rfind(b"PK\x06\x07")  # zip64 end-of-central-directory locator
    if p >= 0:
        off = struct.unpack("<Q", tail[p + 8 : p + 16])[0]
        rec, _ = get(url, {"Range": f"bytes={off}-{off + 55}"})
        cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
    else:
        p = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[p + 12 : p + 20])
    cd, _ = get(url, {"Range": f"bytes={cd_off}-{cd_off + cd_size - 1}"})
    entries = []
    i = 0
    while i < len(cd) and cd[i : i + 4] == b"PK\x01\x02":
        n, m, k = struct.unpack("<HHH", cd[i + 28 : i + 34])
        usize = struct.unpack("<I", cd[i + 24 : i + 28])[0]
        entries.append((cd[i + 46 : i + 46 + n].decode(), usize))
        i += 46 + n + m + k
    return size, entries


def main(version: str, outdir: Path) -> None:
    v = re.escape(version)
    cpu = index_links(f"{BASE}/whl/cpu/torch/")
    default = index_links(f"{BASE}/whl/torch/")
    zips = index_links(f"{BASE}/libtorch/cpu/")
    targets = {
        "wheel-linux_x86_64": (cpu, rf"torch-{v}(%2B|\+)cpu-{PYTHON_TAG}-{PYTHON_TAG}-manylinux_2_28_x86_64\.whl$"),
        "wheel-linux_aarch64": (cpu, rf"torch-{v}(%2B|\+)cpu-{PYTHON_TAG}-{PYTHON_TAG}-manylinux_2_28_aarch64\.whl$"),
        "wheel-macos_arm64": (cpu + default, rf"torch-{v}-{PYTHON_TAG}-{PYTHON_TAG}-macosx_\d+_\d+_arm64\.whl$"),
        "wheel-win_amd64": (cpu, rf"torch-{v}(%2B|\+)cpu-{PYTHON_TAG}-{PYTHON_TAG}-win_amd64\.whl$"),
        "libtorch-linux_x86_64": (zips, rf"libtorch-shared-with-deps-{v}(%2B|\+)cpu\.zip$"),
        "libtorch-macos_arm64": (zips, rf"libtorch-macos-arm64-{v}\.zip$"),
        "libtorch-win_amd64": (zips, rf"libtorch-win-shared-with-deps-{v}(%2B|\+)cpu\.zip$"),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    for tag, (pool, pattern) in targets.items():
        candidates = sorted(u for u in pool if re.search(pattern, u))
        if not candidates:
            print(f"{tag}: not found", file=sys.stderr)
            continue
        url = candidates[-1]
        size, entries = central_directory(url)
        out = outdir / f"{tag}.txt"
        out.write_text("".join(f"{usize}\t{name}\n" for name, usize in entries))
        print(f"{tag}: {url.rsplit('/', 1)[-1]}  {size / 1e6:.0f} MB  {len(entries)} entries -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], Path(sys.argv[2] if len(sys.argv) > 2 else "listings"))
