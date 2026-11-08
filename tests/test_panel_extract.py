import gzip
from pathlib import Path

from izn.io.panel_extract import (docker_cmd, group_by_chrom, load_panel, match_panel, qc_rows,
                                  read_panel_vcf, regions_arg)

VCF = """##fileformat=VCFv4.1
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4
22\t100\trsA\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1\t0|0
22\t200\trsB\tC\tT\t.\tPASS\t.\tGT\t0|0\t0|0\t.|.\t0|0
22\t300\trsC\tG\tA\t.\tPASS\t.\tGT\t0|1\t0|1\t0|1\t0|1
22\t300\trsC2\tG\tT\t.\tPASS\t.\tGT\t0|0\t0|0\t0|0\t0|1
"""


def _vcf(tmp_path: Path) -> Path:
    p = tmp_path / "p.vcf.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(VCF)
    return p


def test_read_and_doses(tmp_path):
    s, r = read_panel_vcf(_vcf(tmp_path))
    assert s == ["S1", "S2", "S3", "S4"]
    assert r[0]["dose"] == [0, 1, 2, 0]
    assert r[1]["dose"] == [0, 0, -1, 0]


def test_match_panel_flags(tmp_path):
    _, r = read_panel_vcf(_vcf(tmp_path))
    panel = [{"rsid": "a", "gene": "G", "chrom": "22", "pos_grch37": 100},
             {"rsid": "b", "gene": "G", "chrom": "22", "pos_grch37": 300},
             {"rsid": "c", "gene": "G", "chrom": "22", "pos_grch37": 999}]
    st = {m["rsid"]: m["status"] for m in match_panel(r, panel)}
    assert st == {"a": "ok", "b": "multi_alt_records", "c": "missing"}


def test_qc_handles_missing_and_groups(tmp_path):
    s, r = read_panel_vcf(_vcf(tmp_path))
    rows = qc_rows(r, s, {"ALL": s, "SUB": ["S1", "S2"]}, min_mac=2)
    assert rows[0]["ALL_alt_af"] == 0.375 and rows[0]["ALL_mac"] == 3 and rows[0]["keep_flag"]
    assert rows[1]["ALL_missing"] == 1 and rows[1]["ALL_mac"] == 0 and not rows[1]["keep_flag"]
    assert rows[0]["SUB_alt_af"] == 0.25


def test_regions_and_command_and_yaml():
    snps = [{"chrom": "22", "pos_grch37": 10}, {"chrom": "22", "pos_grch37": 20}]
    assert regions_arg(snps) == "22:10-10,22:20-20"
    cmd = docker_cmd(Path("/x"), "img", "in.vcf.gz", "22:10-10", "out.vcf.gz")
    assert cmd[:3] == ["docker", "run", "--rm"] and "bcftools norm -m-any" in cmd[-1]
    panel = load_panel(Path("configs/snp_panel.yaml"))
    assert len(panel) == 13 and set(group_by_chrom(panel)) == {"2", "3", "6", "8", "22"}
