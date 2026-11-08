"""Ingest 1000 Genomes VCFs into the object store (see docs for the PowerShell example)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from izn.io.ingest import (BASE_URL, StoreConfig, ensure_bucket, ingest_local, ingest_one,
                           make_client, names_for, remote_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroms", nargs="+", default=["22"])
    ap.add_argument("--bucket", default="vcf-raw")
    ap.add_argument("--staging", type=Path, required=True, help="temporary folder OUTSIDE the repo")
    ap.add_argument("--manifest", type=Path, default=Path("data/external/ingest_manifest.jsonl"))
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--ver", default="v5b", choices=["v5a", "v5b"],
                    help="file release: EBI has v5b; AWS/NCBI mirrors have v5a")
    ap.add_argument("--local-dir", type=Path, default=None,
                    help="folder with files you already downloaded; they are uploaded instead of re-downloaded")
    args = ap.parse_args()

    cfg = StoreConfig(os.environ["S3_ENDPOINT"], os.environ["S3_ACCESS_KEY"], os.environ["S3_SECRET_KEY"])
    client = make_client(cfg)
    ensure_bucket(client, args.bucket)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    for name in names_for(args.chroms, ver=args.ver):
        local = args.local_dir / name if args.local_dir else None
        if local is not None and local.exists():
            try:
                expected = remote_size(f"{args.base_url}/{name}")
            except Exception:
                expected = None
                print(f"warning: could not check remote size for {name}; verify the file yourself")
            status = ingest_local(client, args.bucket, local, args.manifest, expected)
        else:
            status = ingest_one(client, args.bucket, name, args.base_url, args.staging, args.manifest)
        print(f"{status:9s} {name}")


if __name__ == "__main__":
    main()