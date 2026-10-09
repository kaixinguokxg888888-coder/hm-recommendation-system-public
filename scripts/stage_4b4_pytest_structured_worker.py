from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest


def _clean_reason(value: str) -> str:
    text = value.strip()
    for prefix in ("Skipped: ", "XFailed: ", "XPASS "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return " ".join(text.split())


class StructuredResultPlugin:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.nodeids: list[str] = []
        self.reports: dict[str, list[Any]] = {}
        self.collection_errors: list[dict[str, str]] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.nodeids = [item.nodeid.replace("\\", "/") for item in session.items]

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collection_errors.append({
                "nodeid": report.nodeid.replace("\\", "/"),
                "reason": "collection failed",
            })

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.reports.setdefault(report.nodeid.replace("\\", "/"), []).append(report)

    def _location_and_reason(self, report: pytest.TestReport) -> tuple[str, str]:
        longrepr = report.longrepr
        if isinstance(longrepr, tuple) and len(longrepr) == 3:
            raw_path, line, reason = longrepr
            path = Path(str(raw_path))
            try:
                relative = path.resolve().relative_to(self.root).as_posix()
            except ValueError:
                relative = Path(report.nodeid.split("::", 1)[0]).as_posix()
            return f"{relative}:{int(line)}", _clean_reason(str(reason))
        reason = str(getattr(report, "wasxfail", "") or longrepr or "")
        return Path(report.nodeid.split("::", 1)[0]).as_posix(), _clean_reason(reason)

    def result(self, return_code: int) -> dict[str, Any]:
        counts = {
            "collected": len(self.nodeids),
            "passed": 0,
            "failed": 0,
            "errors": len(self.collection_errors),
            "skipped": 0,
            "xfailed": 0,
            "xpassed": 0,
        }
        skipped: list[dict[str, Any]] = []
        for nodeid in self.nodeids:
            reports = self.reports.get(nodeid, [])
            xpass = next((report for report in reports if report.passed and getattr(report, "wasxfail", None)), None)
            xfail = next((report for report in reports if report.skipped and getattr(report, "wasxfail", None)), None)
            normal_skip = next((report for report in reports if report.skipped and not getattr(report, "wasxfail", None)), None)
            setup_or_teardown_failure = next(
                (report for report in reports if report.failed and report.when in {"setup", "teardown"}),
                None,
            )
            call_failure = next((report for report in reports if report.failed and report.when == "call"), None)
            call_pass = next((report for report in reports if report.passed and report.when == "call"), None)
            if xpass is not None:
                counts["xpassed"] += 1
            elif xfail is not None:
                counts["xfailed"] += 1
            elif normal_skip is not None:
                counts["skipped"] += 1
                location, reason = self._location_and_reason(normal_skip)
                skipped.append({
                    "nodeid": nodeid,
                    "reason": reason,
                    "code_location": location,
                })
            elif setup_or_teardown_failure is not None:
                counts["errors"] += 1
            elif call_failure is not None:
                counts["failed"] += 1
            elif call_pass is not None:
                counts["passed"] += 1
            else:
                counts["errors"] += 1
        counts["outcome_total"] = sum(
            counts[name] for name in ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
        )
        return {
            "status": "PASS" if return_code == 0 else "FAIL",
            "return_code": int(return_code),
            "counts": counts,
            "skipped": skipped,
            "collection_errors": self.collection_errors,
            "structured_result_version": "pytest_hook_report_v1",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pytest and persist structured counts and skip details")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if "-ra" not in pytest_args:
        pytest_args.append("-ra")
    root = Path.cwd().resolve()
    # 以 scripts/*.py 入口启动时，Python 默认只把 scripts/ 放入 sys.path；
    # 全量测试中存在对 scripts.* 的正式导入，因此必须显式恢复仓库根的导入语义。
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    plugin = StructuredResultPlugin(root)
    return_code = int(pytest.main(pytest_args, plugins=[plugin]))
    payload = plugin.result(return_code)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, args.output)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
