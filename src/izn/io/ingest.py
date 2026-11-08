"""Resumable download of 1000 Genomes files into S3-compatible object storage.

The raw VCFs are never kept in the project folder: each file is downloaded to a
temporary staging directory (resumable, for slow connections), uploaded to the
object store, verified by size, and the staging copy is deleted.
Works with any S3-compatible backend (MinIO, SeaweedFS, Garage, AWS S3).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import boto3
import requests
from boto3.s3.transfer import TransferConfig
from botocore.client import BaseClient
from botocore.exceptions import ClientError

BASE_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
VCF_TEMPLATE = "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_{ver}.20130502.genotypes.vcf.gz"
CHUNK = 1024 * 1024          # hashing block
DL_CHUNK = 64 * 1024         # download read size (small so .part grows visibly)
PROGRESS_EVERY = 5 * 1024 * 1024


@dataclass(frozen=True)
class StoreConfig:
    endpoint: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"


def make_client(cfg: StoreConfig) -> BaseClient:
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        region_name=cfg.region,
    )


def ensure_bucket(client: BaseClient, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def object_size(client: BaseClient, bucket: str, key: str) -> int | None:
    try:
        return int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    except ClientError:
        return None


def remote_size(url: str, timeout: int = 60) -> int:
    r = requests.head(url, allow_redirects=True, timeout=timeout)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def download_resumable(url: str, dest: Path, expected: int, retries: int = 20,
                       timeout: int = 60, backoff: float = 5.0,
                       verbose: bool = False) -> None:
    """Download with HTTP Range resume; keeps a .part file until the size matches."""
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have == expected:
            break
        if have > expected:
            part.unlink()
            have = 0
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 200 and have:
                    have = 0
                    part.unlink()
                elif r.status_code not in (200, 206):
                    r.raise_for_status()
                with part.open("ab" if have else "wb") as fh:
                    t0, got, last = time.time(), have, have
                    for chunk in r.iter_content(DL_CHUNK):
                        if chunk:
                            fh.write(chunk)
                            fh.flush()
                            got += len(chunk)
                            if verbose and got - last >= PROGRESS_EVERY:
                                last = got
                                rate = (got - have) / 1024 / max(time.time() - t0, 1e-6)
                                print(f"  {dest.name}: {got/1e6:.1f}/{expected/1e6:.1f} MB "
                                      f"({rate:.0f} KB/s)", flush=True)
        except (requests.RequestException, OSError):
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(backoff)
            continue
    if part.stat().st_size != expected:
        raise RuntimeError(f"size mismatch for {dest.name}")
    part.replace(dest)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def ingest_one(client: BaseClient, bucket: str, name: str, base_url: str,
               staging: Path, manifest: Path, timeout: int = 60) -> str:
    """Return 'skipped' or 'uploaded'. Raises on any verification failure."""
    url = f"{base_url}/{name}"
    size = remote_size(url, timeout)
    if object_size(client, bucket, name) == size:
        return "skipped"
    local = staging / name
    download_resumable(url, local, size, timeout=timeout, verbose=True)
    digest = sha256_file(local)
    client.upload_file(str(local), bucket, name,
                       Config=TransferConfig(multipart_threshold=64 * 1024 * 1024,
                                             multipart_chunksize=64 * 1024 * 1024))
    stored = object_size(client, bucket, name)
    if stored != size:
        raise RuntimeError(f"upload verification failed for {name}: {stored} != {size}")
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": name, "bytes": size, "sha256": digest,
                             "source": url, "bucket": bucket}) + "\n")
    local.unlink()
    return "uploaded"


def ingest_local(client: BaseClient, bucket: str, path: Path, manifest: Path,
                 expected_size: int | None = None) -> str:
    """Upload a file you already downloaded (e.g. chr22) without re-downloading it."""
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"{path.name}: local size {size} != expected {expected_size} (incomplete download?)")
    if object_size(client, bucket, path.name) == size:
        return "skipped"
    digest = sha256_file(path)
    client.upload_file(str(path), bucket, path.name,
                       Config=TransferConfig(multipart_threshold=64 * 1024 * 1024,
                                             multipart_chunksize=64 * 1024 * 1024))
    stored = object_size(client, bucket, path.name)
    if stored != size:
        raise RuntimeError(f"upload verification failed for {path.name}: {stored} != {size}")
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": path.name, "bytes": size, "sha256": digest,
                             "source": "local-file", "bucket": bucket}) + "\n")
    return "uploaded"


def names_for(chroms: list[str], with_index: bool = True, ver: str = "v5b") -> list[str]:
    out: list[str] = []
    for c in chroms:
        n = VCF_TEMPLATE.format(chrom=c, ver=ver)
        out.append(n)
        if with_index:
            out.append(n + ".tbi")
    return out