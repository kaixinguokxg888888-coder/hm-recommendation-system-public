from __future__ import annotations


def user_segment(history_interactions: int) -> str:
    if history_interactions < 0:
        raise ValueError("history_interactions 不得为负")
    if history_interactions == 0:
        return "cold"
    if history_interactions <= 4:
        return "short_history"
    if history_interactions <= 20:
        return "warm"
    return "active"


def positive_count_bucket(value: int) -> str:
    if value < 1:
        raise ValueError("positive count 必须至少为 1")
    if value <= 2:
        return str(value)
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    return "10+"


def population_membership(history_interactions: int, positive_count: int) -> dict[str, bool]:
    if positive_count < 1:
        raise ValueError("评估用户必须至少有一个标签正样本")
    return {
        "all_label_users": True,
        "evaluable_warm_users": history_interactions > 0,
        "cold_history_label_users": history_interactions == 0,
    }
