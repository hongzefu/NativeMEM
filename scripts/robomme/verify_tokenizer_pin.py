"""按 tokenizer_pin.json 核对本地 Stage 1 权重目录（计划 G2）。

两档：
  cheap：文件集合相同、逐文件字节数相同、小文件 git blob id 相同、LFS 大文件首尾各 1 MiB 的 blake2b
         只记录（挡不住「保持长度改中间字节」，不作通过依据，只做快速前置）。
  full ：在 cheap 基础上逐文件全量 sha256 与 pin 里 HF 服务端给出的 LFS sha256 比对（小文件比 git blob id）。
末行：TOKENIZER_PIN=PASS|FAIL level=<cheap|full> files=<n> sha=<pin manifest sha256 前 8 位> revision=<40 位>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_tokenizer_pin import canonical_sha256  # noqa: E402


def git_blob_id(path: pathlib.Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=str(HERE / "tokenizer_pin.json"))
    ap.add_argument("--params-dir", required=True)
    ap.add_argument("--level", choices=["cheap", "full"], default="cheap")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    pin = json.loads(pathlib.Path(args.pin).read_text())
    revision = pin.get("revision", "")
    sha8 = (pin.get("manifest_sha256") or "")[:8]
    fails: list[str] = []

    def fail_line(reason: str) -> int:
        print(f"TOKENIZER_PIN=FAIL level={args.level} reason={reason} files={pin.get('num_files')} sha={sha8} revision={revision}")
        return 1

    if canonical_sha256(pin) != pin.get("manifest_sha256"):
        return fail_line("pin_tampered")
    if len(revision) != 40:
        return fail_line("revision_not_40")

    root = pathlib.Path(args.params_dir)
    expected = {e["path"]: e for e in pin["files"]}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if set(expected) != actual:
        missing = sorted(set(expected) - actual)[:5]
        extra = sorted(actual - set(expected))[:5]
        return fail_line(f"file_set_mismatch missing={missing} extra={extra}")

    for rel, e in expected.items():
        p = root / rel
        if p.stat().st_size != e["size"]:
            fails.append(f"size:{rel}")
        elif "git_blob_id" in e and git_blob_id(p) != e["git_blob_id"]:
            fails.append(f"blob:{rel}")
    if fails:
        return fail_line(f"cheap_mismatch n={len(fails)} first={fails[0]}")

    if args.level == "full":
        lfs = [(rel, e) for rel, e in expected.items() if "lfs_sha256" in e]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda item: (item[0], sha256_file(root / item[0]) == item[1]["lfs_sha256"]), lfs))
        bad = [rel for rel, ok in results if not ok]
        if bad:
            return fail_line(f"sha256_mismatch n={len(bad)} first={bad[0]}")

    print(f"TOKENIZER_PIN=PASS level={args.level} files={len(expected)} sha={sha8} revision={revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
