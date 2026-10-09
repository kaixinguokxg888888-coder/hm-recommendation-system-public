from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ranking_model.validation_selection import canonical_fingerprint
from reranking.config import Config, VARIANTS
from reranking.evaluation import relevance_guard, select_config
from reranking.metadata import build_metadata

ROOT = Path(__file__).resolve().parents[1]


def test_preregistration_exact_loss_budget_and_matrix():
    config = Config.load(ROOT / "configs/stage_5_reranking.yaml")
    assert config.raw["selection"]["r0_loss_budget_relative"] == 0.005
    assert tuple(v["id"] for v in config.raw["variants"]) == VARIANTS
    assert config.raw["selection"]["r3_min_unique_garment_gain_vs_r2"] == 0.10
    assert config.raw["selection"]["r4_min_tail_count_gain_vs_r3"] == 0.01
    assert config.raw["selection"]["post_results_policy_search_allowed"] is False


def test_no_real_stage5_test_results_or_execution_interface():
    assert not (ROOT/"artifacts/stage_5/private/test_results").exists()
    raw = json.loads((ROOT/"configs/stage_5_reranking.yaml").read_text(encoding="utf-8"))
    assert all("test_14d" not in v for v in raw["upstream"].values())
    source = (ROOT/"scripts/run_stage_5_reranking.py").read_text(encoding="utf-8")
    assert '"test_evaluation"' not in source


def test_metadata_projection_exact_helper_and_no_lost_keys(tmp_path):
    paths = {}
    fixtures = {
        "articles":pa.table({"article_id":["synthetic-a","synthetic-b","synthetic-c"],"product_code":[1,1,2]}),
        "article_key_map":pa.table({"article_id":["synthetic-a","synthetic-b","synthetic-c"],"article_key":[1,2,3]}),
        "canonical_articles":pa.table({"article_key":[1,2,3],"article_garment_group_code":[7,7,8]}),
        "support":pa.table({"article_key":[1,2],"interaction_count":[1,10],"latest_date":[date(2020,8,25)]*2})}
    for name,table in fixtures.items():
        path=tmp_path/(name+".parquet");pq.write_table(table,path);paths[name]=path.name
    (tmp_path/"src/data_foundation").mkdir(parents=True)
    (tmp_path/"src/data_foundation/item_state.py").write_text("synthetic-helper-binding")
    cfg=Config(tmp_path,{"upstream":paths,"snapshot_id":"validation_14d","as_of_date":"2020-08-25"})
    table, manifest = build_metadata(cfg)
    assert table["article_key"].to_pylist()==[1,2,3]
    assert table["article_garment_group_code"].to_pylist()==[7,7,8]
    assert table["popularity_bucket"].to_pylist()==["mid","head","zero_history"]
    assert manifest["seen_item_count"]==2 and (manifest["p50"],manifest["p90"])==(1,10)


def test_guard_checks_ci_lower_not_only_point():
    base={"map_at_12":0.025,"ndcg_at_12":0.04}
    good={m:{"delta_mean":0.0,"ci_lower":0.0} for m in base}
    assert relevance_guard(good,base,0.005)
    bad={m:{"delta_mean":0.0,"ci_lower":-0.001} for m in base}
    assert not relevance_guard(bad,base,0.005)


def test_selection_falls_back_without_rescue_search():
    s={v:{"map_at_12":0.025,"ndcg_at_12":0.04,"after_product_violation":0.1,"after_unique_product":10.0,
          "after_unique_garment":4.0,"after_garment_violation":0.2,"after_tail_count":0.1} for v in VARIANTS}
    pairs=[(v,"R0") for v in VARIANTS[1:]]+[(v,"R3") for v in VARIANTS[4:]]
    comparisons={pair:{m:{"delta_mean":-0.001,"ci_lower":-0.002} for m in ["map_at_12","ndcg_at_12"]} for pair in pairs}
    config=Config.load(ROOT/"configs/stage_5_reranking.yaml")
    assert select_config(s,comparisons,config.raw["selection"])[0]=="R1"


def test_protected_upstream_hash_audit_when_present():
    path=ROOT/"artifacts/stage_5/upstream_protection_after.json"
    if path.exists():
        audit=json.loads(path.read_text(encoding="utf-8"))
        assert audit["mismatch_count"]==0 and audit["checked_file_count"]>=359
    else:
        audit=json.loads((ROOT/"artifacts/stage_5/upstream_protection_before.json").read_text(encoding="utf-8"))
        assert audit["mismatch_count"]==0


def test_checkpoint_identity_and_outputs_detect_mutation(tmp_path):
    from reranking.runner import Runner
    cfg=Config.load(ROOT/"configs/stage_5_reranking.yaml")
    raw=json.loads(json.dumps(cfg.raw));raw["paths"]={k:str(tmp_path.relative_to(ROOT)/k) for k in raw["paths"]} if tmp_path.is_relative_to(ROOT) else raw["paths"]
    # 使用薄 runner 的独立临时根，证明只复用完整一致的成功结果。
    raw["paths"]={"public":"public","private":"private","checkpoints":"checkpoints","tmp":"tmp"}
    root=tmp_path; (root/"public").mkdir(); (root/"public/upstream_protection_manifest.csv").write_text("path,size_bytes,sha256\n")
    (root/"src/reranking").mkdir(parents=True);(root/"scripts").mkdir()
    (root/"scripts/run_stage_5_reranking.py").write_text("synthetic")
    runner=Runner(Config(root,raw));inp=root/"input";inp.write_text("one");out=root/"output"
    calls=[]
    def work():
        calls.append(1);out.write_text("valid");return {"count":1}
    runner.checkpoint("fixture",[inp],[out],work)
    assert runner.checkpoint("fixture",[inp],[out],work)["reused"] is True
    out.write_text("mutation");runner.checkpoint("fixture",[inp],[out],work)
    assert len(calls)==2


@pytest.mark.parametrize("name,content,allowed", [
    ("stats.json", '{"frozen_absolute_rss_cap_bytes":3210988748}', True),
    ("stats.csv", "peak_rss_bytes\n3210988748\n", True),
    ("stats.md", "RSS: 3210988748 bytes.", True),
    ("sample.json", '{"sample":"0000001234"}', False),
    ("sample.csv", "sample\n0000001234\n", False),
    ("sample.md", 'Sample: "0000001234".', False),
    ("raw.json", '{"article_id":1}', False),
    ("path.json", '{"path":"X:/synthetic"}', False),
])
def test_public_scanner_distinguishes_resource_numbers_from_id_strings(tmp_path, name, content, allowed):
    from reranking.runner import Runner
    public = tmp_path / "public"
    public.mkdir()
    (public / name).write_text(content, encoding="utf-8")
    runner = Runner.__new__(Runner)
    runner.cfg = Config(tmp_path, {"paths": {"public": "public"}})
    if allowed:
        assert runner.scan_public()["violation_count"] == 0
    else:
        with pytest.raises(ValueError, match="privacy/path"):
            runner.scan_public()
