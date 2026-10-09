from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from .schema import model_feature_names


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def category_code_expression(source: str, alias: str) -> str:
    column = f"a.{quote_identifier(source)}"
    return (
        f"CASE WHEN {column} IS NULL THEN 0 ELSE "
        f"dense_rank() OVER (ORDER BY {column} NULLS LAST)+1 END::INTEGER AS {quote_identifier(alias)}"
    )


def query_digest_select(
    schema_rows: Sequence[dict[str, object]], schema_fingerprint: str,
    *, source_glob: str,
) -> str:
    names = ["snapshot_id", "customer_key", "article_key", *model_feature_names(schema_rows)]
    fields = ",".join(f"f.{quote_identifier(name)}" for name in names)
    return f"""
      WITH row_hash AS (
        SELECT snapshot_id,customer_key,candidate_rank,
          hash('ranking-feature-row-left-v1',{fields}) h1,
          hash('ranking-feature-row-right-v1',{fields}) h2
        FROM read_parquet('{source_glob}') f
      )
      SELECT snapshot_id,customer_key,count(*)::INTEGER row_count,
        sha256('ranking-feature-query-v1' || chr(0) || '{schema_fingerprint}' || chr(0) ||
               string_agg(lpad(to_hex(h1),16,'0') || lpad(to_hex(h2),16,'0'),'' ORDER BY candidate_rank)) query_digest
      FROM row_hash GROUP BY snapshot_id,customer_key ORDER BY customer_key
    """


def feature_select_columns(canonical_fields: Iterable[dict[str, str]]) -> list[str]:
    columns = [
        "c.snapshot_id", "c.as_of_date", "c.customer_key", "c.article_key",
        "c.candidate_contract_version", "c.candidate_protocol_version", "c.primary_source",
        "'ranking_feature_pipeline_snapshot_safe_v1'::VARCHAR feature_pipeline_version",
        "'{schema_fingerprint}'::VARCHAR feature_schema_fingerprint",
        "coalesce(u.interaction_count,0)::BIGINT user_history_interaction_count",
        "coalesce(u.distinct_item_count,0)::BIGINT user_history_distinct_item_count",
        "date_diff('day',u.latest_date,c.as_of_date)::INTEGER user_recency_days",
        "coalesce(u.recent_28d_count,0)::BIGINT user_recent_interaction_count_28d",
        "CASE WHEN coalesce(u.interaction_count,0)=0 THEN NULL ELSE 1.0-u.distinct_item_count::DOUBLE/u.interaction_count END::DOUBLE user_repeat_ratio",
        "CASE WHEN coalesce(u.interaction_count,0)=0 THEN 'cold' WHEN u.interaction_count<=4 THEN 'short_history' WHEN u.interaction_count<=20 THEN 'warm' ELSE 'active' END::VARCHAR user_segment",
        "coalesce(i.interaction_count,0)::BIGINT item_history_interaction_count",
        "coalesce(i.distinct_buyer_count,0)::BIGINT item_history_distinct_buyer_count",
        "coalesce(i.recent_7d_count,0)::BIGINT item_recent_interaction_count_7d",
        "date_diff('day',i.latest_date,c.as_of_date)::INTEGER item_recency_days",
        "coalesce(ui.interaction_count,0)::BIGINT ui_history_interaction_count",
        "(coalesce(ui.interaction_count,0)>0)::BOOLEAN ui_seen_before",
        "date_diff('day',ui.latest_date,c.as_of_date)::INTEGER ui_recency_days",
        "c.candidate_rank", "c.source_count",
        "c.from_repurchase", "c.from_itemcf", "c.from_two_tower", "c.from_trend", "c.from_popularity",
        "c.repurchase_rank", "c.repurchase_score", "c.itemcf_rank", "c.itemcf_score",
        "c.two_tower_rank", "c.two_tower_score", "c.trend_rank", "c.trend_score",
        "c.popularity_rank", "c.popularity_score",
    ]
    for index, field in enumerate(canonical_fields):
        semantic = field["semantic_field"]
        code = f"article_{semantic}_code"
        columns.append(f"a.{quote_identifier(code)}")
        columns.append(
            f"CASE WHEN coalesce(u.interaction_count,0)=0 OR a.{quote_identifier(code)}=0 THEN NULL "
            f"ELSE coalesce(uc{index}.interaction_count,0)::DOUBLE/u.interaction_count END::DOUBLE "
            f"{quote_identifier(f'user_{semantic}_interaction_share')}"
        )
    return columns
