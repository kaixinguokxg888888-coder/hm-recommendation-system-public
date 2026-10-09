from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .stage1a import Stage1AConfig, Stage1AEngine
from .stage1b import activity_bucket, read_csv, save_group_output, write_csv


def _prepare_codes(engine: Stage1AEngine, include_attributes: bool = False) -> None:
    engine.execute(
        "CREATE TEMP TABLE _customer_codes AS "
        "SELECT customer_id, row_number() OVER ()::BIGINT customer_code FROM customers"
    )
    attributes = (
        ", product_code, product_type_no, department_no, garment_group_no"
        if include_attributes
        else ""
    )
    engine.execute(
        "CREATE TEMP TABLE _article_codes AS "
        "SELECT article_id, row_number() OVER ()::BIGINT article_code"
        f"{attributes} FROM articles"
    )


def _prepare_coded_transactions(
    engine: Stage1AEngine, include_attributes: bool = False
) -> None:
    _prepare_codes(engine, include_attributes)
    attributes = (
        ", a.product_code, a.product_type_no, a.department_no, a.garment_group_no"
        if include_attributes
        else ""
    )
    engine.execute(
        "CREATE TEMP TABLE _tx AS "
        "SELECT try_cast(t.t_dat AS DATE) t_dat, c.customer_code, a.article_code, "
        f"t.price, t.sales_channel_id{attributes} "
        "FROM transactions t "
        "JOIN _customer_codes c ON t.customer_id=c.customer_id "
        "JOIN _article_codes a ON t.article_id=a.article_id"
    )


def _quantiles(engine: Stage1AEngine, table: str, field: str) -> dict[str, float]:
    values = engine.row(
        f"SELECT quantile_cont({field}, [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]) "
        f"FROM {table}"
    )[0]
    return dict(zip(["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"], values))


def group_a_temporal_activity(
    engine: Stage1AEngine, config: Stage1AConfig, output: Path
) -> tuple[dict[str, Any], list[Path]]:
    _prepare_coded_transactions(engine)
    end_date = engine.row("SELECT max(t_dat) FROM _tx")[0]
    rows: list[dict[str, Any]] = []
    for granularity, expression in [
        ("day", "t_dat"),
        ("week", "date_trunc('week', t_dat)"),
        ("month", "date_trunc('month', t_dat)"),
    ]:
        base = engine.rows(
            f"SELECT {expression} period_start, count(*) raw_rows, "
            "count(*) FILTER (WHERE sales_channel_id=1) channel_1_rows, "
            "count(*) FILTER (WHERE sales_channel_id=2) channel_2_rows "
            "FROM _tx GROUP BY period_start ORDER BY period_start"
        )
        users = dict(
            engine.rows(
                f"SELECT {expression} period_start,count(DISTINCT customer_code) "
                "FROM _tx GROUP BY period_start"
            )
        )
        items = dict(
            engine.rows(
                f"SELECT {expression} period_start,count(DISTINCT article_code) "
                "FROM _tx GROUP BY period_start"
            )
        )
        binary = dict(
            engine.rows(
                f"SELECT period_start,count(*) FROM (SELECT {expression} period_start,"
                "customer_code,article_code FROM _tx GROUP BY ALL) "
                "GROUP BY period_start"
            )
        )
        for row in base:
            rows.append(
                {
                    "granularity": granularity,
                    "period_start": str(row[0]),
                    "raw_rows_A": int(row[1]),
                    "active_users": int(users[row[0]]),
                    "active_items": int(items[row[0]]),
                    "channel_1_rows": int(row[2]),
                    "channel_2_rows": int(row[3]),
                    "binary_interactions_B": int(binary[row[0]]),
                    "daily_strength_sum_C": int(row[1]),
                    "same_day_excess_A_minus_B": int(row[1] - binary[row[0]]),
                    "days_to_data_end": (
                        (end_date - row[0]).days if granularity == "day" else ""
                    ),
                    "execution_scope": "FULL_SCAN",
                }
            )
    path = config.artifact_dir / "temporal_activity.csv"
    write_csv(path, rows)
    daily = [row for row in rows if row["granularity"] == "day"]
    monthly = [row for row in rows if row["granularity"] == "month"]
    payload = {
        "min_date": daily[0]["period_start"],
        "max_date": daily[-1]["period_start"],
        "day_count": len(daily),
        "raw_rows": sum(row["raw_rows_A"] for row in daily),
        "binary_interactions": sum(row["binary_interactions_B"] for row in daily),
        "same_day_excess": sum(row["same_day_excess_A_minus_B"] for row in daily),
        "min_month_rows": min(row["raw_rows_A"] for row in monthly),
        "max_month_rows": max(row["raw_rows_A"] for row in monthly),
        "first_month_rows": monthly[0]["raw_rows_A"],
        "last_month_rows": monthly[-1]["raw_rows_A"],
    }
    save_group_output(output, "group_a_temporal_activity", payload)
    return payload, [path, output]


def _bucket_sql(field: str) -> str:
    return (
        f"CASE WHEN {field}=1 THEN '1' WHEN {field}=2 THEN '2' "
        f"WHEN {field}<=5 THEN '3-5' WHEN {field}<=10 THEN '6-10' "
        f"WHEN {field}<=20 THEN '11-20' WHEN {field}<=50 THEN '21-50' "
        f"WHEN {field}<=100 THEN '51-100' ELSE '100+' END"
    )


def group_b_user_activity(
    engine: Stage1AEngine, config: Stage1AConfig, output: Path
) -> tuple[dict[str, Any], list[Path]]:
    _prepare_codes(engine)
    end_date = engine.row("SELECT max(try_cast(t_dat AS DATE)) FROM transactions")[0]
    engine.execute(
        "CREATE TEMP TABLE _user_days AS "
        "SELECT c.customer_code, try_cast(t.t_dat AS DATE) t_dat "
        "FROM transactions t JOIN _customer_codes c ON t.customer_id=c.customer_id "
        "GROUP BY c.customer_code,t_dat"
    )
    engine.execute(
        "CREATE TEMP TABLE _user_gaps AS "
        "WITH gaps AS (SELECT customer_code,t_dat,"
        "date_diff('day',lag(t_dat) OVER(PARTITION BY customer_code ORDER BY t_dat),t_dat) gap "
        "FROM _user_days) SELECT customer_code,avg(gap) avg_gap FROM gaps GROUP BY customer_code"
    )
    engine.execute(
        "CREATE TEMP TABLE _user_profile AS "
        "SELECT c.customer_code, count(*) raw_A, "
        "count(DISTINCT (t.t_dat,a.article_code)) binary_B, "
        "count(DISTINCT a.article_code) distinct_items, "
        "count(DISTINCT t.t_dat) active_days, "
        "min(try_cast(t.t_dat AS DATE)) first_date,max(try_cast(t.t_dat AS DATE)) last_date "
        "FROM transactions t "
        "JOIN _customer_codes c ON t.customer_id=c.customer_id "
        "JOIN _article_codes a ON t.article_id=a.article_id "
        "GROUP BY c.customer_code"
    )
    total_users, total_raw, total_binary, total_distinct_sum = engine.row(
        "SELECT count(*),sum(raw_A),sum(binary_B),sum(distinct_items) FROM _user_profile"
    )
    bucket_rows: list[dict[str, Any]] = []
    for scope, field in [("A", "raw_A"), ("B", "binary_B"), ("C", "raw_A")]:
        values = engine.rows(
            f"SELECT {_bucket_sql(field)} bucket,count(*) users,sum({field}) interactions,"
            "sum(distinct_items) distinct_item_sum FROM _user_profile GROUP BY bucket"
        )
        order = ["1", "2", "3-5", "6-10", "11-20", "21-50", "51-100", "100+"]
        mapped = {row[0]: row for row in values}
        denominator = total_raw if scope in {"A", "C"} else total_binary
        for label in order:
            row = mapped.get(label, (label, 0, 0, 0))
            bucket_rows.append(
                {
                    "feedback_scope": scope,
                    "bucket": label,
                    "user_count": int(row[1]),
                    "user_share": int(row[1]) / total_users,
                    "interaction_count": int(row[2]),
                    "interaction_coverage": int(row[2]) / denominator,
                    "distinct_item_sum": int(row[3]),
                    "distinct_item_coverage": int(row[3]) / total_distinct_sum,
                    "execution_scope": "FULL_SCAN",
                }
            )
    bucket_path = config.artifact_dir / "user_activity_buckets.csv"
    write_csv(bucket_path, bucket_rows)
    quantile_rows = []
    for scope, field in [("A", "raw_A"), ("B", "binary_B"), ("C", "raw_A")]:
        quantile_rows.append(
            {
                "feedback_scope": scope,
                "metric": "sequence_length",
                **_quantiles(engine, "_user_profile", field),
                "execution_scope": "FULL_SCAN",
            }
        )
    for metric, field, table in [
        ("distinct_articles", "distinct_items", "_user_profile"),
        ("active_days", "active_days", "_user_profile"),
        ("lifecycle_days", "date_diff('day',first_date,last_date)", "_user_profile"),
        ("average_active_day_gap", "avg_gap", "_user_gaps"),
    ]:
        quantile_rows.append(
            {
                "feedback_scope": "aggregate",
                "metric": metric,
                **_quantiles(engine, table, field),
                "execution_scope": "FULL_SCAN",
            }
        )
    quantile_path = config.artifact_dir / "user_sequence_quantiles.csv"
    write_csv(quantile_path, quantile_rows)
    thresholds = {
        str(threshold): int(
            engine.row(f"SELECT count(*) FROM _user_profile WHERE binary_B>={threshold}")[0]
        )
        for threshold in config.raw["analysis"]["sequence_thresholds"]
    }
    one_behavior = int(engine.row("SELECT count(*) FROM _user_profile WHERE binary_B=1")[0])
    recent_gap_quantiles = engine.row(
        f"SELECT quantile_cont(date_diff('day',last_date,DATE '{end_date}'),"
        "[0.25,0.5,0.75,0.9,0.99]) FROM _user_profile"
    )[0]
    payload = {
        "users_with_transactions": int(total_users),
        "one_binary_interaction_users": one_behavior,
        "one_binary_interaction_user_rate": one_behavior / total_users,
        "users_at_least_binary_interactions": thresholds,
        "coverage_rates": {
            key: value / total_users for key, value in thresholds.items()
        },
        "raw_interactions": int(total_raw),
        "binary_interactions": int(total_binary),
        "same_day_excess": int(total_raw - total_binary),
        "recent_gap_quantiles_days": dict(
            zip(["p25", "p50", "p75", "p90", "p99"], recent_gap_quantiles)
        ),
    }
    save_group_output(output, "group_b_user_activity", payload)
    return payload, [bucket_path, quantile_path, output]


def group_c_item_long_tail(
    engine: Stage1AEngine, config: Stage1AConfig, output: Path
) -> tuple[dict[str, Any], list[Path]]:
    _prepare_codes(engine)
    engine.execute(
        "CREATE TEMP TABLE _item_profile AS "
        "SELECT a.article_code,count(*) raw_A,"
        "count(DISTINCT (t.t_dat,c.customer_code)) binary_B,"
        "count(DISTINCT c.customer_code) distinct_customers,"
        "count(DISTINCT t.t_dat) active_days,"
        "min(try_cast(t.t_dat AS DATE)) first_date,max(try_cast(t.t_dat AS DATE)) last_date "
        "FROM transactions t JOIN _article_codes a ON t.article_id=a.article_id "
        "JOIN _customer_codes c ON t.customer_id=c.customer_id GROUP BY a.article_code"
    )
    item_count, total_raw = map(
        int, engine.row("SELECT count(*),sum(raw_A) FROM _item_profile")
    )
    total_articles = int(engine.row("SELECT count(*) FROM articles")[0])
    bucket_rows = []
    for scope, field in [("A", "raw_A"), ("B", "binary_B"), ("C", "raw_A")]:
        values = engine.rows(
            f"SELECT {_bucket_sql(field)} bucket,count(*) items,sum({field}) interactions "
            f"FROM _item_profile GROUP BY bucket"
        )
        total_interactions = int(
            engine.row(f"SELECT sum({field}) FROM _item_profile")[0]
        )
        for row in values:
            bucket_rows.append(
                {
                    "feedback_scope": scope,
                    "bucket": row[0],
                    "item_count": int(row[1]),
                    "item_share": int(row[1]) / item_count,
                    "interaction_count": int(row[2]),
                    "interaction_coverage": int(row[2]) / total_interactions,
                    "execution_scope": "FULL_SCAN",
                }
            )
    bucket_path = config.artifact_dir / "item_popularity_buckets.csv"
    write_csv(bucket_path, bucket_rows)
    shares = {}
    for fraction in [0.01, 0.05, 0.10, 0.20]:
        limit = max(1, math.ceil(item_count * fraction))
        shares[f"top_{int(fraction*100)}pct"] = float(
            engine.row(
                f"SELECT sum(raw_A)/{total_raw}::DOUBLE FROM "
                f"(SELECT raw_A FROM _item_profile ORDER BY raw_A DESC LIMIT {limit})"
            )[0]
        )
    gini = float(
        engine.row(
            "WITH ranked AS (SELECT raw_A,row_number() OVER(ORDER BY raw_A) rn,"
            "count(*) OVER() n,sum(raw_A) OVER() total FROM _item_profile) "
            "SELECT 2*sum(rn*raw_A)/(max(n)*max(total))-"
            "(max(n)+1)::DOUBLE/max(n) FROM ranked"
        )[0]
    )
    hhi = float(
        engine.row(
            f"SELECT sum(power(raw_A/{total_raw}::DOUBLE,2)) FROM _item_profile"
        )[0]
    )
    long_tail_rows = [
        {
            "feedback_scope": "A",
            "transacted_items": item_count,
            "untransacted_articles": total_articles - item_count,
            "gini": gini,
            "hhi": hhi,
            **shares,
            **_quantiles(engine, "_item_profile", "raw_A"),
            "execution_scope": "FULL_SCAN",
        },
        {
            "feedback_scope": "B",
            "transacted_items": item_count,
            "untransacted_articles": total_articles - item_count,
            "gini": "",
            "hhi": "",
            **{key: "" for key in shares},
            **_quantiles(engine, "_item_profile", "binary_B"),
            "execution_scope": "FULL_SCAN",
        },
    ]
    metrics_path = config.artifact_dir / "item_long_tail_metrics.csv"
    write_csv(metrics_path, long_tail_rows)
    payload = {
        "transacted_items": item_count,
        "untransacted_articles": total_articles - item_count,
        "raw_interactions": total_raw,
        "gini_raw": gini,
        "hhi_raw": hhi,
        "top_shares_raw": shares,
        "raw_popularity_quantiles": _quantiles(engine, "_item_profile", "raw_A"),
        "binary_popularity_quantiles": _quantiles(engine, "_item_profile", "binary_B"),
    }
    save_group_output(output, "group_c_item_long_tail", payload)
    return payload, [bucket_path, metrics_path, output]


def group_d_repurchase_stability(
    engine: Stage1AEngine, config: Stage1AConfig, output: Path
) -> tuple[dict[str, Any], list[Path]]:
    _prepare_coded_transactions(engine, include_attributes=True)
    engine.execute(
        "CREATE TEMP TABLE _ui_day AS SELECT customer_code,article_code,t_dat,"
        "count(*) daily_count FROM _tx GROUP BY customer_code,article_code,t_dat"
    )
    raw_total = int(engine.row("SELECT count(*) FROM _tx")[0])
    binary_total = int(engine.row("SELECT count(*) FROM _ui_day")[0])
    engine.execute(
        "CREATE TEMP TABLE _ui AS SELECT customer_code,article_code,"
        "sum(daily_count) raw_count,count(*) active_days,min(t_dat) first_date,max(t_dat) last_date "
        "FROM _ui_day GROUP BY customer_code,article_code"
    )
    users = int(engine.row("SELECT count(DISTINCT customer_code) FROM _tx")[0])
    repeat = engine.row(
        "SELECT count(*) FILTER(WHERE raw_count>1),"
        "count(DISTINCT customer_code) FILTER(WHERE raw_count>1),"
        "sum(raw_count-1) FILTER(WHERE raw_count>1),"
        "count(*) FILTER(WHERE active_days>1),"
        "count(DISTINCT customer_code) FILTER(WHERE active_days>1),"
        "sum(active_days-1) FILTER(WHERE active_days>1) FROM _ui"
    )
    intervals = engine.row(
        "WITH d AS (SELECT customer_code,article_code,t_dat,"
        "date_diff('day',lag(t_dat) OVER(PARTITION BY customer_code,article_code ORDER BY t_dat),t_dat) gap "
        "FROM _ui_day) SELECT quantile_cont(gap,[0.01,0.25,0.5,0.75,0.9,0.99]) "
        "FROM d WHERE gap IS NOT NULL"
    )[0]
    engine.execute(
        "CREATE TEMP TABLE _up AS SELECT customer_code,product_code,count(*) raw_count,"
        "count(DISTINCT article_code) distinct_articles,count(DISTINCT t_dat) active_days "
        "FROM _tx GROUP BY customer_code,product_code"
    )
    product = engine.row(
        "SELECT count(*) FILTER(WHERE raw_count>1),"
        "count(DISTINCT customer_code) FILTER(WHERE raw_count>1),"
        "count(*) FILTER(WHERE distinct_articles>1),"
        "count(DISTINCT customer_code) FILTER(WHERE distinct_articles>1) FROM _up"
    )
    hierarchy_rows = []
    for field in ["product_type_no", "department_no", "garment_group_no"]:
        row = engine.row(
            f"WITH x AS (SELECT customer_code,{field},count(*) n FROM _tx "
            f"WHERE {field} IS NOT NULL GROUP BY customer_code,{field}) "
            "SELECT count(*) FILTER(WHERE n>1),"
            "count(DISTINCT customer_code) FILTER(WHERE n>1),max(n) FROM x"
        )
        hierarchy_rows.append(
            {
                "metric": f"repeat_preference_{field}",
                "group_count": int(row[0]),
                "user_count": int(row[1]),
                "max_count": int(row[2]),
                "execution_scope": "FULL_SCAN",
            }
        )
    end_date = engine.row("SELECT max(t_dat) FROM _tx")[0]
    cutoff = end_date - __import__("datetime").timedelta(
        days=int(config.raw["analysis"]["preference_recent_window_days"]) - 1
    )
    overlap = engine.row(
        f"WITH h AS (SELECT DISTINCT customer_code,department_no FROM _tx "
        f"WHERE t_dat<DATE '{cutoff}' AND department_no IS NOT NULL),"
        f"r AS (SELECT DISTINCT customer_code,department_no FROM _tx "
        f"WHERE t_dat>=DATE '{cutoff}' AND department_no IS NOT NULL),"
        "hc AS (SELECT customer_code,count(*) n FROM h GROUP BY customer_code),"
        "rc AS (SELECT customer_code,count(*) n FROM r GROUP BY customer_code),"
        "ic AS (SELECT h.customer_code,count(*) n FROM h JOIN r USING(customer_code,department_no) "
        "GROUP BY h.customer_code),"
        "j AS (SELECT hc.customer_code,coalesce(ic.n,0)::DOUBLE/(hc.n+rc.n-coalesce(ic.n,0)) score "
        "FROM hc JOIN rc USING(customer_code) LEFT JOIN ic USING(customer_code)) "
        "SELECT count(*),avg(score),quantile_cont(score,[0.1,0.25,0.5,0.75,0.9]) FROM j"
    )
    profile = {
        "raw_rows_A": raw_total,
        "binary_user_item_days_B": binary_total,
        "daily_strength_sum_C": raw_total,
        "same_day_excess": raw_total - binary_total,
        "same_article_repeat_groups": int(repeat[0]),
        "users_with_same_article_repeat": int(repeat[1]),
        "user_repeat_rate": int(repeat[1]) / users,
        "repeat_excess_rows": int(repeat[2]),
        "repeat_excess_row_rate": int(repeat[2]) / raw_total,
        "cross_day_repeat_groups": int(repeat[3]),
        "users_with_cross_day_repeat": int(repeat[4]),
        "cross_day_excess_days": int(repeat[5]),
        "repurchase_gap_quantiles_days": dict(
            zip(["p01", "p25", "p50", "p75", "p90", "p99"], intervals)
        ),
        "same_product_repeat_groups": int(product[0]),
        "users_with_product_repeat": int(product[1]),
        "same_product_different_article_groups": int(product[2]),
        "users_with_same_product_different_article": int(product[3]),
        "recent_history_department_overlap": {
            "users_in_both_windows": int(overlap[0]),
            "mean_jaccard": float(overlap[1]),
            "quantiles": dict(
                zip(["p10", "p25", "p50", "p75", "p90"], overlap[2])
            ),
        },
    }
    rows = [
        {
            "metric": key,
            "value": json.dumps(value, ensure_ascii=False),
            "feedback_scope": (
                "A/B/C comparison"
                if key in {"raw_rows_A", "binary_user_item_days_B", "daily_strength_sum_C", "same_day_excess"}
                else "aggregate"
            ),
            "execution_scope": "FULL_SCAN",
        }
        for key, value in profile.items()
    ]
    rows.extend(hierarchy_rows)
    path = config.artifact_dir / "repurchase_metrics.csv"
    write_csv(path, rows)
    save_group_output(output, "group_d_repurchase_stability", profile)
    return profile, [path, output]


def _window_cold_metrics(
    engine: Stage1AEngine, cutoff: str
) -> dict[str, Any]:
    row = engine.row(
        f"WITH tu AS (SELECT DISTINCT customer_code FROM _tx WHERE t_dat<DATE '{cutoff}'),"
        f"ti AS (SELECT DISTINCT article_code FROM _tx WHERE t_dat<DATE '{cutoff}'),"
        f"test AS (SELECT x.*,tu.customer_code seen_u,ti.article_code seen_i FROM _tx x "
        f"LEFT JOIN tu ON x.customer_code=tu.customer_code "
        f"LEFT JOIN ti ON x.article_code=ti.article_code WHERE x.t_dat>=DATE '{cutoff}') "
        "SELECT count(*),count(DISTINCT customer_code),count(DISTINCT article_code),"
        "count(DISTINCT customer_code) FILTER(WHERE seen_u IS NOT NULL),"
        "count(DISTINCT article_code) FILTER(WHERE seen_i IS NOT NULL),"
        "count(*) FILTER(WHERE seen_u IS NULL),count(*) FILTER(WHERE seen_i IS NULL),"
        "count(*) FILTER(WHERE seen_u IS NULL AND seen_i IS NULL),"
        "count(*) FILTER(WHERE seen_u IS NOT NULL AND seen_i IS NOT NULL) FROM test"
    )
    test_rows, test_users, test_items = map(int, row[:3])
    seen_users, seen_items = int(row[3]), int(row[4])
    return {
        "test_rows": test_rows,
        "test_users": test_users,
        "seen_test_users": seen_users,
        "new_test_users": test_users - seen_users,
        "user_cold_start_rate": (test_users - seen_users) / test_users,
        "test_items": test_items,
        "seen_test_items": seen_items,
        "new_test_items": test_items - seen_items,
        "item_cold_start_rate": (test_items - seen_items) / test_items,
        "cold_user_rows": int(row[5]),
        "cold_user_row_rate": int(row[5]) / test_rows,
        "cold_item_rows": int(row[6]),
        "cold_item_row_rate": int(row[6]) / test_rows,
        "cold_both_rows": int(row[7]),
        "cold_both_row_rate": int(row[7]) / test_rows,
        "warm_user_item_rows": int(row[8]),
        "warm_candidate_upper_bound": int(row[8]) / test_rows,
    }


def group_e_cold_start(
    engine: Stage1AEngine, config: Stage1AConfig, output: Path
) -> tuple[dict[str, Any], list[Path]]:
    _prepare_coded_transactions(engine)
    end_date = engine.row("SELECT max(t_dat) FROM _tx")[0]
    rows = []
    payload = {}
    for window in config.raw["analysis"]["evaluation_windows_days"]:
        cutoff = end_date - __import__("datetime").timedelta(days=int(window) - 1)
        metrics = _window_cold_metrics(engine, str(cutoff))
        payload[str(window)] = {"cutoff": str(cutoff), **metrics}
        rows.append(
            {
                "window_days": window,
                "train_end": str(cutoff - __import__("datetime").timedelta(days=1)),
                "test_start": str(cutoff),
                "test_end": str(end_date),
                **metrics,
                "execution_scope": "TIME_WINDOW_FULL_SCAN",
            }
        )
    path = config.artifact_dir / "cold_start_by_window.csv"
    write_csv(path, rows)
    save_group_output(output, "group_e_cold_start", payload)
    return payload, [path, output]


def group_f_time_split(
    engine: Stage1AEngine, config: Stage1AConfig, output: Path
) -> tuple[dict[str, Any], list[Path]]:
    _prepare_coded_transactions(engine)
    start_date, end_date = engine.row("SELECT min(t_dat),max(t_dat) FROM _tx")
    rows = []
    payload = {}
    for window in config.raw["analysis"]["evaluation_windows_days"]:
        cutoff = end_date - __import__("datetime").timedelta(days=int(window) - 1)
        suffix = str(window)
        engine.execute(
            f"CREATE TEMP TABLE _train_user_{suffix} AS SELECT customer_code,count(*) history "
            f"FROM _tx WHERE t_dat<DATE '{cutoff}' GROUP BY customer_code"
        )
        engine.execute(
            f"CREATE TEMP TABLE _train_item_{suffix} AS SELECT DISTINCT article_code "
            f"FROM _tx WHERE t_dat<DATE '{cutoff}'"
        )
        engine.execute(
            f"CREATE TEMP TABLE _test_user_{suffix} AS SELECT customer_code,count(*) positives "
            f"FROM _tx WHERE t_dat>=DATE '{cutoff}' GROUP BY customer_code"
        )
        base = engine.row(
            f"SELECT "
            f"(SELECT count(*) FROM _tx WHERE t_dat<DATE '{cutoff}'),"
            f"(SELECT count(*) FROM _tx WHERE t_dat>=DATE '{cutoff}'),"
            f"(SELECT count(*) FROM _train_user_{suffix}),"
            f"(SELECT count(*) FROM _test_user_{suffix}),"
            f"(SELECT count(DISTINCT article_code) FROM _tx WHERE t_dat<DATE '{cutoff}'),"
            f"(SELECT count(DISTINCT article_code) FROM _tx WHERE t_dat>=DATE '{cutoff}')"
        )
        user_metrics = engine.row(
            f"SELECT count(*) FILTER(WHERE tr.customer_code IS NOT NULL),"
            f"count(*) FILTER(WHERE positives=1),"
            f"count(*) FILTER(WHERE tr.customer_code IS NOT NULL AND tr.history<5),"
            f"count(*) FILTER(WHERE tr.customer_code IS NOT NULL AND tr.history<10),"
            f"quantile_cont(positives,[0.25,0.5,0.75,0.9,0.99]) "
            f"FROM _test_user_{suffix} te LEFT JOIN _train_user_{suffix} tr USING(customer_code)"
        )
        coverage = engine.row(
            f"SELECT count(*) FILTER(WHERE u.customer_code IS NOT NULL),"
            f"count(*) FILTER(WHERE i.article_code IS NOT NULL),"
            f"count(*) FILTER(WHERE u.customer_code IS NOT NULL AND i.article_code IS NOT NULL) "
            f"FROM _tx x LEFT JOIN _train_user_{suffix} u USING(customer_code) "
            f"LEFT JOIN _train_item_{suffix} i USING(article_code) WHERE x.t_dat>=DATE '{cutoff}'"
        )
        test_users = int(base[3])
        test_rows = int(base[1])
        metrics = {
            "train_rows": int(base[0]),
            "test_rows": test_rows,
            "train_users": int(base[2]),
            "test_users": test_users,
            "train_items": int(base[4]),
            "test_items": int(base[5]),
            "evaluable_users": int(user_metrics[0]),
            "test_user_train_visibility": int(user_metrics[0]) / test_users,
            "one_test_positive_users": int(user_metrics[1]),
            "one_test_positive_user_rate": int(user_metrics[1]) / test_users,
            "evaluable_users_history_lt5": int(user_metrics[2]),
            "evaluable_users_history_lt10": int(user_metrics[3]),
            "test_positive_quantiles": dict(
                zip(["p25", "p50", "p75", "p90", "p99"], user_metrics[4])
            ),
            "warm_user_row_rate": int(coverage[0]) / test_rows,
            "warm_item_row_rate": int(coverage[1]) / test_rows,
            "warm_user_item_candidate_upper_bound": int(coverage[2]) / test_rows,
        }
        payload[str(window)] = {"cutoff": str(cutoff), **metrics}
        rows.append(
            {
                "window_days": window,
                "train_start": str(start_date),
                "train_end": str(cutoff - __import__("datetime").timedelta(days=1)),
                "test_start": str(cutoff),
                "test_end": str(end_date),
                **{
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, dict)
                    else value
                    for key, value in metrics.items()
                },
                "random_split": False,
                "execution_scope": "TIME_WINDOW_FULL_SCAN",
            }
        )
        engine.execute(f"DROP TABLE _train_user_{suffix}")
        engine.execute(f"DROP TABLE _train_item_{suffix}")
        engine.execute(f"DROP TABLE _test_user_{suffix}")
    path = config.artifact_dir / "time_split_comparison.csv"
    write_csv(path, rows)
    save_group_output(output, "group_f_time_split", payload)
    return payload, [path, output]


def group_g_model_evidence(
    engine: Stage1AEngine, config: Stage1AConfig, output: Path
) -> tuple[dict[str, Any], list[Path]]:
    del engine
    user = json.loads((config.group_dir / "group_b_user_activity.json").read_text(encoding="utf-8"))["payload"]
    item = json.loads((config.group_dir / "group_c_item_long_tail.json").read_text(encoding="utf-8"))["payload"]
    repurchase = json.loads((config.group_dir / "group_d_repurchase_stability.json").read_text(encoding="utf-8"))["payload"]
    cold = json.loads((config.group_dir / "group_e_cold_start.json").read_text(encoding="utf-8"))["payload"]
    split = json.loads((config.group_dir / "group_f_time_split.json").read_text(encoding="utf-8"))["payload"]
    coverage10 = user["coverage_rates"]["10"]
    warm_bounds = ",".join(
        f"{split[key]['warm_user_item_candidate_upper_bound']:.4f}"
        for key in ["7", "14", "28"]
    )
    evidence = [
        ("热门召回", f"Top 1% share={item['top_shares_raw']['top_1pct']:.4f}; Gini={item['gini_raw']:.4f}", "长尾与时变流行度风险", "all users, especially cold users", "low", "high", "yes", "V1 baseline", "离线增益尚未验证"),
        ("时间趋势召回", "734-day temporal series and channel/month aggregates available", "需要严格只用切分点前窗口", "all users", "low-medium", "high", "yes", "V1 candidate", "最佳趋势窗口未知"),
        ("历史复购召回", f"user repeat rate={repurchase['user_repeat_rate']:.4f}", "重复记录不等于真实 quantity", "users with history", "low", "high", "yes", "V1 candidate", "A/B/C 最佳口径未知"),
        ("ItemCF", f"binary interactions={user['binary_interactions']}; active-user coverage>=10={coverage10:.4f}", "短序列用户共现稀疏", "warm users", "medium", "high", "yes", "V1 candidate", "实际 recall/NDCG 尚未知"),
        ("商品内容/属性召回", f"untransacted articles={item['untransacted_articles']}; article attributes available", "属性质量和语义相似度未验证", "cold users/items", "medium", "high", "yes", "V1 cold-start component", "属性权重未知"),
        ("矩阵分解", f"users={user['users_with_transactions']}; items={item['transacted_items']}", "时间漂移和冷启动", "warm users/items", "medium", "medium-high", "yes", "baseline/enhancement", "最佳隐式反馈权重未知"),
        ("双塔", f"warm candidate upper bounds 7/14/28={warm_bounds}", "需要负采样、特征和线上检索工程", "warm and feature-covered users/items", "high", "high", "experiment", "enhancement experiment", "统计证据不能证明模型收益"),
        ("序列模型", f"binary sequence >=10 coverage={coverage10:.4f}", "大量短序列用户不适用；尚无训练样本", "active users only", "high", "high", "experiment", "subgroup experiment", "GRU4Rec/DIN/Transformer 优劣不可得出"),
        ("冷启动内容召回", f"28-day cold rates user={cold['28']['user_cold_start_rate']:.4f}, item={cold['28']['item_cold_start_rate']:.4f}", "依赖商品属性，用户侧内容有限", "new users/items", "medium", "high", "yes", "V1 fallback", "真实覆盖增益未知"),
        ("多样性重排", f"item Gini={item['gini_raw']:.4f}", "需要相关性候选和多样性指标", "all served users", "medium", "high", "experiment", "V1 simple rule or later enhancement", "业务多样性目标未知"),
    ]
    rows = [
        {
            "candidate": row[0],
            "data_support_evidence": row[1],
            "data_limitations": row[2],
            "applicable_users": row[3],
            "engineering_cost": row[4],
            "recruiting_demo_value": row[5],
            "recommend_v1": row[6],
            "role": row[7],
            "cannot_conclude": row[8],
            "execution_scope": "AGGREGATE_EVIDENCE",
        }
        for row in evidence
    ]
    path = config.artifact_dir / "model_evidence_matrix.csv"
    write_csv(path, rows)
    payload = {"candidate_count": len(rows), "matrix_path": config.relative(path)}
    save_group_output(output, "group_g_model_evidence", payload)
    return payload, [path, output]


GROUP_FUNCTIONS = {
    "group_a_temporal_activity": group_a_temporal_activity,
    "group_b_user_activity": group_b_user_activity,
    "group_c_item_long_tail": group_c_item_long_tail,
    "group_d_repurchase_stability": group_d_repurchase_stability,
    "group_e_cold_start": group_e_cold_start,
    "group_f_time_split": group_f_time_split,
    "group_g_model_evidence": group_g_model_evidence,
}
