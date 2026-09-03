#!/usr/bin/env python3
"""Analyze logs emitted by ``torch_comm_debug``.

Examples::

    python -m sglang.srt.utils.parse_torch_comm_debug logs/rank*.log
    python -m sglang.srt.utils.parse_torch_comm_debug logs/ --json
    cat rank0.log rank1.log | python -m sglang.srt.utils.parse_torch_comm_debug -

The parser compares the ordered communication operation stream for every rank
and reports calls that did not reach ``after`` (or ended in ``error``).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, TextIO


_LINE_RE = re.compile(
    r"\[NPU-COMM\]\s+"
    r"pid=(?P<pid>\S+)\s+"
    r"seq=(?P<seq>\d+)\s+"
    r"phase=(?P<phase>before|after|error)\s+"
    r"op=(?P<op>\S+)"
    r"(?:\s+(?P<details>.*))?$"
)


@dataclass
class Record:
    source: str
    line: int
    pid: int | None
    seq: int
    phase: str
    op: str
    details: str = ""
    global_rank: int | None = None
    world: int | None = None
    group_ranks: tuple[int, ...] = ()


@dataclass
class Issue:
    kind: str
    message: str
    rank: str | None = None
    source: str | None = None
    line: int | None = None
    seq: int | None = None


@dataclass
class RankReport:
    rank: str
    global_rank: int | None = None
    world: int | None = None
    group_key: str = "unknown"
    group_ranks: list[int] = field(default_factory=list)
    process_ids: list[int] = field(default_factory=list)
    records: int = 0
    calls: int = 0
    completed: int = 0
    pending: int = 0
    errors: int = 0
    issues: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)


def _rank_from_details(details: str) -> str | None:
    match = re.search(r"(?:^|,)rank=(-?\d+)(?:,|$)", details)
    return match.group(1) if match else None


def _int_from_details(details: str, name: str) -> int | None:
    match = re.search(rf"(?:^|,){re.escape(name)}=(-?\d+)(?:,|$)", details)
    return int(match.group(1)) if match else None


def _group_ranks_from_details(details: str) -> tuple[int, ...]:
    match = re.search(r"(?:^|,)group_ranks=([^,\[]+|unknown)(?:,tensors=|$)", details)
    if not match or match.group(1) == "unknown":
        return ()
    try:
        return tuple(int(item) for item in match.group(1).split("|"))
    except ValueError:
        return ()


def _group_key(record: Record) -> str:
    if record.op.startswith("deepep."):
        return "deepep-global"
    if record.group_ranks:
        return "group[" + "|".join(str(item) for item in record.group_ranks) + "]"
    if record.world is not None:
        return f"world={record.world}:unknown-members"
    return "unknown"


def _parse_stream(stream: TextIO, source: str) -> list[Record]:
    records: list[Record] = []
    for line_number, line in enumerate(stream, 1):
        match = _LINE_RE.search(line.rstrip("\n"))
        if not match:
            continue
        pid_text = match.group("pid")
        records.append(
            Record(
                source=source,
                line=line_number,
                pid=int(pid_text) if pid_text.isdigit() else None,
                seq=int(match.group("seq")),
                phase=match.group("phase"),
                op=match.group("op"),
                details=match.group("details") or "",
                global_rank=_int_from_details(match.group("details") or "", "global_rank"),
                world=_int_from_details(match.group("details") or "", "world"),
                group_ranks=_group_ranks_from_details(match.group("details") or ""),
            )
        )
    return records


def _rank_key(record: Record, source_index: int) -> str:
    rank = record.global_rank
    if rank is None:
        rank_text = _rank_from_details(record.details)
        rank = int(rank_text) if rank_text is not None else None
    if rank is not None and rank != -1:
        return f"{_group_key(record)}/rank{rank}"
    if record.pid is not None:
        return f"{_group_key(record)}/pid{record.pid}"
    return f"{_group_key(record)}/source{source_index}"


def _issue(
    report: RankReport,
    kind: str,
    message: str,
    record: Record | None = None,
) -> None:
    report.issues.append(
        Issue(
            kind=kind,
            message=message,
            rank=report.rank,
            source=record.source if record else None,
            line=record.line if record else None,
            seq=record.seq if record else None,
        )
    )


def _warning(
    report: RankReport,
    kind: str,
    message: str,
    record: Record | None = None,
) -> None:
    report.warnings.append(
        Issue(
            kind=kind,
            message=message,
            rank=report.rank,
            source=record.source if record else None,
            line=record.line if record else None,
            seq=record.seq if record else None,
        )
    )


def _build_reports(records_by_rank: dict[str, list[Record]]) -> dict[str, RankReport]:
    reports: dict[str, RankReport] = {}
    for rank, records in records_by_rank.items():
        report = RankReport(rank=rank, records=len(records))
        first_record = records[0]
        report.global_rank = first_record.global_rank
        report.world = first_record.world
        report.group_ranks = list(first_record.group_ranks)
        report.group_key = _group_key(first_record)
        process_ids = sorted({record.pid for record in records if record.pid is not None})
        report.process_ids = process_ids
        if len(process_ids) > 1:
            _warning(
                report,
                "multiple_processes_same_rank",
                f"rank contains multiple pids: {','.join(map(str, process_ids))}; "
                "operation order comparison may be ambiguous",
            )
        # ``seq`` is process-local, so pid is part of the call identity.
        pending: dict[tuple[int | None, int], Record] = {}
        seen_phase: set[tuple[int | None, int, str]] = set()
        previous_seq_by_pid: dict[int | None, int] = {}

        for record in records:
            phase_key = (record.pid, record.seq, record.phase)
            if phase_key in seen_phase:
                _issue(
                    report,
                    "duplicate_phase",
                    f"duplicate {record.phase} for seq={record.seq} op={record.op}",
                    record,
                )
            seen_phase.add(phase_key)

            previous_seq = previous_seq_by_pid.get(record.pid)
            if previous_seq is not None and record.seq < previous_seq:
                _issue(
                    report,
                    "seq_decrease",
                    f"sequence decreased from {previous_seq} to {record.seq}",
                    record,
                )
            previous_seq_by_pid[record.pid] = record.seq

            if record.phase == "before":
                report.calls += 1
                report.operations.append(record.op)
                call_key = (record.pid, record.seq)
                if call_key in pending:
                    _issue(
                        report,
                        "duplicate_before",
                        f"second before for seq={record.seq}; previous op="
                        f"{pending[record.seq].op}, current op={record.op}",
                        record,
                    )
                pending[call_key] = record
                continue

            call_key = (record.pid, record.seq)
            start = pending.get(call_key)
            if start is None:
                _issue(
                    report,
                    "orphan_phase",
                    f"{record.phase} without before for seq={record.seq} "
                    f"op={record.op}",
                    record,
                )
                continue
            if start.op != record.op:
                _issue(
                    report,
                    "op_mismatch_within_rank",
                    f"seq={record.seq} started as {start.op} but ended as {record.op}",
                    record,
                )
            del pending[call_key]
            if record.phase == "after":
                report.completed += 1
            else:
                report.errors += 1
                _issue(
                    report,
                    "communication_error",
                    f"communication failed for seq={record.seq} op={record.op}: "
                    f"{record.details}",
                    record,
                )

        report.pending = len(pending)
        for start in pending.values():
            _issue(
                report,
                "missing_after",
                f"no after/error observed for seq={start.seq} op={start.op}",
                start,
            )
        reports[rank] = report
    return reports


def _compare_operations(
    reports: dict[str, RankReport], *, deepep_world_size: int = 16
) -> list[Issue]:
    issues: list[Issue] = []
    reports_by_group: dict[str, list[RankReport]] = defaultdict(list)
    for report in reports.values():
        reports_by_group[report.group_key].append(report)
    for group_key, group_reports in reports_by_group.items():
        if len(group_reports) < 2:
            continue
        # A rank with multiple process ids contains multiple independent
        # sequence counters.  Do not compare its merged operation list against
        # another rank; the per-pid completion checks above remain valid.
        if any(len(report.process_ids) > 1 for report in group_reports):
            continue
        group_reports.sort(key=lambda report: report.rank)
        reference = group_reports[0]
        for candidate in group_reports[1:]:
            rank = candidate.rank
            common = min(len(reference.operations), len(candidate.operations))
            mismatch_index = next(
                (
                    index
                    for index in range(common)
                    if reference.operations[index] != candidate.operations[index]
                ),
                None,
            )
            if mismatch_index is not None:
                issues.append(
                    Issue(
                        kind="operation_order_mismatch",
                        message=(
                            f"{rank} differs from {reference.rank} in group "
                            f"{group_key} at call #{mismatch_index + 1}: "
                            f"{candidate.operations[mismatch_index]} vs "
                            f"{reference.operations[mismatch_index]}"
                        ),
                        rank=rank,
                    )
                )
            if len(reference.operations) != len(candidate.operations):
                issues.append(
                    Issue(
                        kind="operation_count_mismatch",
                        message=(
                            f"{rank} has {len(candidate.operations)} calls, while "
                            f"{reference.rank} has {len(reference.operations)} in "
                            f"group {group_key}"
                        ),
                        rank=rank,
                    )
                )
    deepep_reports = reports_by_group.get("deepep-global", [])
    if deepep_reports and deepep_world_size > 0:
        present = {
            report.global_rank
            for report in deepep_reports
            if report.global_rank is not None and report.global_rank >= 0
        }
        missing = sorted(set(range(deepep_world_size)) - present)
        if missing:
            issues.append(
                Issue(
                    kind="deepep_missing_ranks",
                    message=(
                        f"DeepEP logs are missing global rank(s) {missing}; "
                        f"expected {deepep_world_size} total rank(s)"
                    ),
                    rank="deepep-global",
                )
            )
    return issues


def analyze(
    paths: Iterable[str], *, deepep_world_size: int = 16
) -> tuple[dict[str, RankReport], list[Issue], int]:
    records_by_rank: dict[str, list[Record]] = defaultdict(list)
    parsed_lines = 0
    for source_index, path in enumerate(paths):
        if path == "-":
            records = _parse_stream(sys.stdin, "<stdin>")
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as stream:
                records = _parse_stream(stream, path)
        parsed_lines += len(records)
        for record in records:
            records_by_rank[_rank_key(record, source_index)].append(record)

    for records in records_by_rank.values():
        records.sort(key=lambda record: (record.seq, record.line, record.phase))
    reports = _build_reports(records_by_rank)
    cross_rank_issues = _compare_operations(
        reports, deepep_world_size=deepep_world_size
    )
    return reports, cross_rank_issues, parsed_lines


def _print_text(
    reports: dict[str, RankReport], cross_rank_issues: list[Issue], parsed_lines: int
) -> None:
    if not parsed_lines:
        print("No [NPU-COMM] records found.")
        return
    print(f"Parsed {parsed_lines} communication records across {len(reports)} rank(s).")
    for rank in sorted(reports):
        report = reports[rank]
        print(
            f"{rank}: calls={report.calls}, completed={report.completed}, "
            f"pending={report.pending}, errors={report.errors}, "
            f"issues={len(report.issues)}, warnings={len(report.warnings)}"
        )
    issues = [issue for report in reports.values() for issue in report.issues]
    issues.extend(cross_rank_issues)
    warnings = [warning for report in reports.values() for warning in report.warnings]
    if not issues:
        suffix = f" ({len(warnings)} warning(s))" if warnings else ""
        print(
            "RESULT: OK - no communication ordering or completion mismatch found."
            + suffix
        )
        for warning in warnings:
            location = ""
            if warning.source is not None:
                location = f" ({warning.source}:{warning.line})"
            print(f"- [warning:{warning.kind}] {warning.message}{location}")
        return
    print(f"RESULT: PROBLEM - found {len(issues)} issue(s).")
    for warning in warnings:
        location = ""
        if warning.source is not None:
            location = f" ({warning.source}:{warning.line})"
        print(f"- [warning:{warning.kind}] {warning.message}{location}")
    for issue in issues:
        location = ""
        if issue.source is not None:
            location = f" ({issue.source}:{issue.line})"
        print(f"- [{issue.kind}] {issue.message}{location}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="+",
        help="log files, directories, shell globs, or '-' for stdin",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit a machine-readable JSON report"
    )
    parser.add_argument(
        "--deepep-world-size",
        type=int,
        default=16,
        help="expected number of global ranks for DeepEP calls (default: 16)",
    )
    args = parser.parse_args(argv)

    expanded: list[str] = []
    for item in args.logs:
        path = Path(item)
        if path.is_dir():
            expanded.extend(str(child) for child in sorted(path.glob("*.log")))
        elif any(char in item for char in "*?["):
            expanded.extend(sorted(glob.glob(item)))
        else:
            expanded.append(item)
    expanded = list(dict.fromkeys(expanded))

    reports, cross_rank_issues, parsed_lines = analyze(
        expanded, deepep_world_size=args.deepep_world_size
    )
    all_issues = [issue for report in reports.values() for issue in report.issues]
    all_issues.extend(cross_rank_issues)
    warnings = [warning for report in reports.values() for warning in report.warnings]
    if args.json:
        payload = {
            "parsed_records": parsed_lines,
            "ranks": {rank: asdict(report) for rank, report in reports.items()},
            "cross_rank_issues": [asdict(issue) for issue in cross_rank_issues],
            "issues": [asdict(issue) for issue in all_issues],
            "warnings": [asdict(warning) for warning in warnings],
            "ok": not all_issues and parsed_lines > 0,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_text(reports, cross_rank_issues, parsed_lines)
    if not parsed_lines:
        return 2
    return 1 if all_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
