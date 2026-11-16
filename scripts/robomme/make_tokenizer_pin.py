"""生成作者发布的 NativeMEM Stage 1 记忆编码器权重的身份锁 tokenizer_pin.json（计划 G2）。

只读访问 HF API：取 Ziyeeee/NativeMEM 当前 40 位 commit sha，以及
checkpoints/mem_tokenizer_pretrain/pretrain/49999/params/ 下每个文件的字节数、LFS sha256（大文件）或
git blob id（小文件）。期望值全部来自 HF 服务端，不现场哈希本地文件（避免自证循环）。
表本身防篡改：顶层 manifest_sha256 = 剔掉该键后 canonical JSON 的 sha256。

用法：
  uv run --no-sync python scripts/robomme/make_tokenizer_pin.py [--out scripts/robomme/tokenizer_pin.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re

REPO_ID = "Ziyeeee/NativeMEM"
REPO_TYPE = "model"
PARAMS_PREFIX = "checkpoints/mem_tokenizer_pretrain/pretrain/49999/params/"
HERE = pathlib.Path(__file__).resolve().parent


def canonical_sha256(obj: dict) -> str:
    body = {k: v for k, v in obj.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "tokenizer_pin.json"))
    ap.add_argument("--revision", default=None, help="默认取 main 当前 commit；给定时必须是 40 位 sha")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, revision=args.revision, files_metadata=True)
    revision = info.sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise SystemExit(f"TOKENIZER_PIN=FAIL reason=bad_revision revision={revision}")

    files = []
    for s in info.siblings:
        if not s.rfilename.startswith(PARAMS_PREFIX):
            continue
        rel = s.rfilename[len(PARAMS_PREFIX):]
        entry = {"path": rel, "size": int(s.size)}
        if s.lfs:
            entry["lfs_sha256"] = s.lfs.sha256
            if int(s.lfs.size) != int(s.size):
                raise SystemExit(f"TOKENIZER_PIN=FAIL reason=lfs_size_mismatch path={rel}")
        else:
            entry["git_blob_id"] = s.blob_id
        files.append(entry)
    if not files:
        raise SystemExit("TOKENIZER_PIN=FAIL reason=no_files")
    files.sort(key=lambda e: e["path"])

    pin = {
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "revision": revision,
        "params_prefix": PARAMS_PREFIX,
        "local_params_dir_default": "checkpoints/mem_tokenizer_pretrain/pretrain/49999/params",
        "num_files": len(files),
        "total_bytes": sum(e["size"] for e in files),
        "files": files,
    }
    pin["manifest_sha256"] = canonical_sha256(pin)
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(pin, indent=2, sort_keys=True) + "\n")
    print(
        f"TOKENIZER_PIN_MADE=PASS files={len(files)} bytes={pin['total_bytes']} "
        f"revision={revision} manifest_sha256={pin['manifest_sha256']} out={out}"
    )


if __name__ == "__main__":
    main()
