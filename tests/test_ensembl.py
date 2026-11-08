from izn.io import ensembl

VARIATION = {"name": "rs000", "minor_allele": "A", "MAF": 0.1, "mappings": [
    {"assembly_name": "GRCh37", "seq_region_name": "HSCHR6_MHC_COX_CTG1", "start": 5, "end": 5, "allele_string": "G/A"},
    {"assembly_name": "GRCh37", "seq_region_name": "22", "start": 1234, "end": 1234, "allele_string": "G/A"},
    {"assembly_name": "GRCh38", "seq_region_name": "22", "start": 9999, "end": 9999, "allele_string": "G/A"}]}


def test_picks_standard_chromosome_grch37_only():
    assert ensembl.grch37_mapping(VARIATION) == {"chrom": "22", "pos": 1234, "end": 1234, "allele_string": "G/A"}


def test_biallelic_snv_check():
    assert ensembl.is_biallelic_snv("G/A")
    assert not ensembl.is_biallelic_snv("G/A/T")
    assert not ensembl.is_biallelic_snv("-/AT")


def test_lookup_uses_gene_overlap(monkeypatch):
    def fake(path, session=None, timeout=30):
        return VARIATION if path.startswith("/variation") else [{"external_name": "TMPRSS6"}]
    monkeypatch.setattr(ensembl, "fetch_json", fake)
    row = ensembl.lookup("rs000")
    assert row["chrom"] == "22" and row["pos_grch37"] == 1234 and row["genes"] == "TMPRSS6" and row["biallelic_snv"]
