from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from data_foundation.item_state import popularity_bucket, popularity_thresholds
from reranking.contracts import Candidate, INPUT_SCHEMA, OUTPUT_SCHEMA, candidates_from_columns, normalize, order_query, output_rows
from reranking.evaluation import diagnostics, metrics
from reranking.greedy import cap_ok, rerank


def candidates(n=30, products=None, garments=None, tails=()):
    return [Candidate(i+1, i+1, -float(i+1), i+1,
                      products[i] if products else i+1, garments[i] if garments else i % 6 + 1, i+1 in tails)
            for i in range(n)]


def article_ids(selection):
    return [r.article for r, _, _ in selection]


def test_no_conflict_exact_rank_identity():
    rows = candidates()
    assert article_ids(rerank(rows, product_cap=2, garment_cap=4)) == list(range(1, 13))


def test_duplicate_article_and_conflicting_metadata_still_dedup():
    rows = candidates()
    rows[1] = Candidate(1, 2, -2.0, 2, 999, 999)
    result = rerank(rows)
    assert len(article_ids(result)) == len(set(article_ids(result))) == 12
    assert 2 not in article_ids(result)
    assert result[0][0].product == 1


def test_product_cap():
    rows = candidates(products=[1]*6 + list(range(2, 26)))
    result = rerank(rows, product_cap=2)
    assert cap_ok([r for r, _, _ in result], 2, None)
    assert article_ids(result)[:3] == [1, 2, 7]


def test_garment_cap():
    rows = candidates(30, garments=[1]*8 + [2]*8 + [3]*14)
    result = rerank(rows, garment_cap=4)
    assert cap_ok([r for r, _, _ in result], None, 4)


def test_simultaneous_caps():
    rows = candidates(40, products=[i//4 for i in range(40)], garments=[i//10 for i in range(40)])
    result = rerank(rows, product_cap=2, garment_cap=4)
    assert len(result) == 12 and cap_ok([r for r, _, _ in result], 2, 4)


def test_relax_garment_level1():
    rows = candidates(30, garments=[1]*15 + [2]*15)
    result = rerank(rows, product_cap=2, garment_cap=4)
    assert len(result) == 12
    assert max(level for _, _, level in result) == 1
    assert "relax_garment" in {reason for _, reason, _ in result}


def test_relax_product_level2():
    rows = candidates(40, products=[i//10 for i in range(40)])
    result = rerank(rows, product_cap=2)
    assert len(result) == 12
    assert max(level for _, _, level in result) == 2


def test_unique_final_fill_level3():
    rows = candidates(30, products=[1]*30, garments=[1]*30)
    result = rerank(rows, product_cap=2, garment_cap=4)
    assert len(result) == 12
    assert max(level for _, _, level in result) == 3
    assert "final_fill" in {reason for _, reason, _ in result}


@pytest.mark.parametrize("n", [0, 1, 11])
def test_candidate_less_than_k(n):
    result = rerank(candidates(n), product_cap=2, garment_cap=4)
    assert len(result) == n


def test_unique_candidates_less_than_k():
    rows = [Candidate(i%5, i+1, -float(i), i+1, 1, 1) for i in range(30)]
    result = rerank(rows, product_cap=2, garment_cap=4)
    assert len(result) == 5 and len(set(article_ids(result))) == 5
    assert diagnostics(rows, result, None)["insufficient_unique_candidates"] == 1


def test_tail_disabled_and_enabled():
    rows = candidates(tails=(15,))
    off = rerank(rows, product_cap=2, garment_cap=4)
    on = rerank(rows, product_cap=2, garment_cap=4, tail_window=20)
    assert article_ids(off) == list(range(1, 13))
    assert article_ids(on) == list(range(1, 12)) + [15]
    assert on[-1][1] == "tail_exploration"


def test_tail_outside_rank_guard():
    rows = candidates(60, tails=(51,))
    assert article_ids(rerank(rows, tail_window=50)) == list(range(1, 13))


def test_tail_not_forced_if_caps_conflict_or_list_relaxed():
    rows = candidates(products=[1, 1] + list(range(2, 14)) + [1] + list(range(20, 35)), tails=(15,))
    assert not any(reason == "tail_exploration" for _, reason, _ in rerank(rows, product_cap=2, tail_window=20))
    relaxed = candidates(30, garments=[1]*30, tails=(15,))
    assert not any(reason == "tail_exploration" for _, reason, _ in rerank(relaxed, garment_cap=4, tail_window=20))


def test_natural_tail_preserves_list_and_multiple_natural_tails():
    rows = candidates(tails=(2, 3, 15))
    assert article_ids(rerank(rows, tail_window=20)) == list(range(1, 13))


def test_determinism_and_partition_boundary():
    users = [candidates(tails=(14,)), candidates(products=[1]*8 + list(range(2,24)))]
    once = [rerank(r, product_cap=2, garment_cap=4, tail_window=20) for r in users]
    batches = [rerank(r, product_cap=2, garment_cap=4, tail_window=20) for chunk in [users[:1],users[1:]] for r in chunk]
    assert once == batches


def table_fixture(test_schema=False):
    rows = {"snapshot_id": ["test_14d" if test_schema else "validation_14d"]*2,
            "customer_key": [32,32], "article_key": [1,2], "candidate_rank": [1,2],
            "raw_score": [2.0,1.0], "predicted_rank": [1,2]}
    if test_schema:
        rows.update(version_id=["L1_FINAL_REFIT_600"]*2, partition_id=[0,0])
    else:
        rows.update(model_id=["L1_600"]*2, feature_set=["F1"]*2, model_fingerprint=["synthetic"]*2)
    return pa.table(rows)


def test_validation_normalization():
    result = normalize(table_fixture(), 0)
    assert result.schema == INPUT_SCHEMA and result["partition_id"].to_pylist() == [0,0]


def test_future_test_schema_synthetic_only():
    result = normalize(table_fixture(True), 0, snapshot="test_14d")
    assert result.schema == INPUT_SCHEMA
    assert "version_id" not in result.column_names


@pytest.mark.parametrize("field,value", [("raw_score", [np.nan,1.0]), ("customer_key", [33,33]), ("predicted_rank", [None,2])])
def test_normalization_rejects_drift(field,value):
    table = table_fixture(); table = table.set_column(table.schema.get_field_index(field), field, pa.array(value))
    with pytest.raises(ValueError):
        normalize(table, 0)


def test_metadata_missing_candidate_rejected():
    columns = normalize(table_fixture(),0).to_pydict()
    with pytest.raises(ValueError,match="metadata join"):
        candidates_from_columns(columns, {1:(1,1,4,True)},0,2)


def test_output_schema_continuity_unique_original_order():
    selected = rerank(candidates(),product_cap=2,garment_cap=4)
    table = pa.Table.from_pylist(output_rows(32,0,selected),schema=OUTPUT_SCHEMA)
    assert table.schema == OUTPUT_SCHEMA
    assert table["rerank_position"].to_pylist() == list(range(1,13))
    assert len(set(table["article_key"].to_pylist())) == 12


def test_input_order_restored_and_frozen_score_tie_rejected():
    rows = candidates(500)
    assert order_query(list(reversed(rows)),formal=True) == rows
    rows[1] = Candidate(2,2,5.0,2,2,2)
    with pytest.raises(ValueError,match="tie-break"):
        order_query(rows,formal=True)


def test_missing_attributes_not_shared_group():
    rows = [Candidate(i+1,i+1,-float(i),i+1,None,None) for i in range(30)]
    assert article_ids(rerank(rows,product_cap=2,garment_cap=4)) == list(range(1,13))


def test_relevance_reuses_all_positive_denominator_and_all_zero():
    selected = rerank(candidates())
    value = metrics(selected,{1,3,999,1000},4)
    assert value["map_at_12"] == pytest.approx((1+2/3)/4)
    assert value["recall_at_12"] == 0.5
    assert value["ndcg_at_12"] == pytest.approx((1+1/math.log2(4))/sum(1/math.log2(i+1) for i in range(1,5)))
    assert metrics(selected,{999,1000},2) == {m:0.0 for m in value}


def test_existing_tail_boundaries_population_zero_history():
    p50,p90 = popularity_thresholds([1,2,3,4,5,6,7,8,9,10])
    assert (p50,p90) == (5,9)
    assert [popularity_bucket(n,p50,p90) for n in [0,4,5,8,9]] == ["zero_history","tail","mid","mid","head"]
    with pytest.raises(ValueError):
        popularity_thresholds([])
