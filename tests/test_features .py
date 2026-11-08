from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from izn.features.panel_features import build_table, read_pop_map

PANEL = [{"rsid": "rsA", "chrom": "22", "pos_grch37": 100}, {"rsid": "rsB", "chrom": "3", "pos_grch37": 5}]


def _files(tmp_path: Path):
    g = tmp_path / "g.tsv"
    g.write_text("sample\t22:100:A>G\t3:5:C>T\nS1\t0\t2\nS2\t1\t-1\nS3\t2\t0\n", encoding="utf-8")
    pm = tmp_path / "p.panel"
    pm.write_text("sample\tpop\tsuper_pop\tgender\nS1\tYRI\tAFR\tmale\nS2\tLWK\tAFR\tfemale\n", encoding="utf-8")
    return g, pm


def test_build_table_order_pops_and_parquet(tmp_path):
    g, pm = _files(tmp_path)
    cols, meta = build_table(g, ["S2", "S1"], PANEL, ["rsB", "rsA"], read_pop_map(pm))
    assert cols["sample"] == ["S2", "S1"] and cols["pop"] == ["LWK", "YRI"]
    assert cols["rsA"] == [1, 0] and cols["rsB"] == [-1, 2]
    assert meta["rsA"] == {"ref": "A", "alt": "G"}
    pq.write_table(pa.table(cols), tmp_path / "t.parquet")
    assert pq.read_table(tmp_path / "t.parquet").num_rows == 2


def test_missing_sample_or_feature_raises(tmp_path):
    g, pm = _files(tmp_path)
    with pytest.raises(ValueError):
        build_table(g, ["S1", "ZZ"], PANEL, ["rsA"], {})
    with pytest.raises(ValueError):
        build_table(g, ["S1"], PANEL, ["rsX"], {})


def test_scenarios_file_is_consistent():
    cfg = yaml.safe_load(Path("configs/scenarios.yaml").read_text(encoding="utf-8"))
    panel = {p["rsid"]: p for p in yaml.safe_load(Path("configs/snp_panel.yaml").read_text(encoding="utf-8"))["panel"]}
    assert len(cfg["features"]) == 13 and not set(cfg["features"]) & set(cfg["excluded_from_features"])
    for sc in cfg["scenarios"].values():
        if "weights" in sc:
            for trait, w in sc["weights"].items():
                assert set(w) == set(sc[trait])
        for trait in ("iron", "zinc"):
            for rsid, allele in sc[trait].items():
                assert panel[rsid]["risk_allele"] == allele
                assert panel[rsid]["trait"].startswith(trait)
                assert rsid in cfg["features"]
