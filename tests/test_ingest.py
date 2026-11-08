import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from moto.server import ThreadedMotoServer

from izn.io.ingest import StoreConfig, download_resumable, ensure_bucket, ingest_one, make_client

DATA = os.urandom(3 * 1024 * 1024 + 123)


class RangeHandler(BaseHTTPRequestHandler):
    fail_once = {"done": False}

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(DATA)))
        self.end_headers()

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
        body = DATA[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not rng and not RangeHandler.fail_once["done"]:
            RangeHandler.fail_once["done"] = True
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)


@pytest.fixture(scope="module")
def http_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(scope="module")
def s3():
    srv = ThreadedMotoServer(port=5055)
    srv.start()
    yield make_client(StoreConfig("http://127.0.0.1:5055", "test", "test"))
    srv.stop()


def test_resume_after_dropped_connection(tmp_path: Path, http_server: str):
    RangeHandler.fail_once["done"] = False
    dest = tmp_path / "f.bin"
    download_resumable(f"{http_server}/f.bin", dest, len(DATA), backoff=0.1)
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == hashlib.sha256(DATA).hexdigest()


def test_ingest_uploads_verifies_and_cleans_staging(tmp_path: Path, http_server: str, s3):
    RangeHandler.fail_once["done"] = True
    ensure_bucket(s3, "vcf-raw")
    staging, manifest = tmp_path / "stage", tmp_path / "m.jsonl"
    assert ingest_one(s3, "vcf-raw", "x.vcf.gz", http_server, staging, manifest) == "uploaded"
    assert s3.get_object(Bucket="vcf-raw", Key="x.vcf.gz")["Body"].read() == DATA
    assert not (staging / "x.vcf.gz").exists()
    rec = json.loads(manifest.read_text().splitlines()[0])
    assert rec["bytes"] == len(DATA)
    assert ingest_one(s3, "vcf-raw", "x.vcf.gz", http_server, staging, manifest) == "skipped"


def test_ingest_local_file_checks_size_and_uploads(tmp_path: Path, s3):
    from izn.io.ingest import ingest_local
    ensure_bucket(s3, "vcf-raw")
    f = tmp_path / "chr22.vcf.gz"
    f.write_bytes(b"abc" * 1000)
    manifest = tmp_path / "m.jsonl"
    with pytest.raises(RuntimeError):
        ingest_local(s3, "vcf-raw", f, manifest, expected_size=5)
    assert ingest_local(s3, "vcf-raw", f, manifest, expected_size=3000) == "uploaded"
    assert ingest_local(s3, "vcf-raw", f, manifest, expected_size=3000) == "skipped"
