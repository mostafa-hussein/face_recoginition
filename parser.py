#!/usr/bin/env python3
"""
Protocol-log analyzer (v2)
———————————
• Success  = block that starts with "----Starting ROS----"
             and contains at least one "--- RUNNING MATCH ----"
• Failure  = block that starts with "----Starting ROS----"
             but ends (next start or EOF) with **no** run-marker.
• Protocol name = most recent "high_level_domain_<Name> started"
                  seen **inside** the same block.
• Date pulled from filename pattern: log_Y25_M05_D19 → 2025-05-19
"""
from __future__ import annotations
import argparse, pathlib, re, csv, json
from typing import List, Dict
from pathlib import Path

LOG_STEM     = re.compile(r"log_Y(\d{2})_M(\d{2})_D(\d{2})")
PROTO_LINE   = re.compile(r"high_level_domain_(\w+)\s+started")
START_MARKER = "----Starting ROS----"
RUN_MARKER   = "--- RUNNING MATCH ----"

def analyze_file(path: pathlib.Path) -> Dict[str, str | int]:
    m = LOG_STEM.search(path.stem)
    if not m:
        return {}
    date = f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"

    success = failure = 0
    failed_protocols: list[str] = []

    waiting      = False          # Are we inside an open “attempt”?
    found_run    = False          # Did this attempt succeed?
    last_proto   = None           # Last protocol name seen in this attempt

    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()

        if START_MARKER in line:
            # close any previous attempt
            if waiting:
                if found_run:
                    success += 1
                else:
                    failure += 1
                    if last_proto:
                        failed_protocols.append(last_proto)
            # open a new attempt
            waiting, found_run, last_proto = True, False, None
            continue

        if waiting:
            m2 = PROTO_LINE.search(line)
            if m2:
                last_proto = m2.group(1)
            if RUN_MARKER in line:
                found_run = True

    # finalize last attempt (if file ended without new START)
    if waiting:
        if found_run:
            success += 1
        else:
            failure += 1
            if last_proto:
                failed_protocols.append(last_proto)

    return {
        "date": date,
        "success": success,
        "failure": failure,
        "failed_protocols": ", ".join(sorted(set(failed_protocols))),
    }

def analyze_folder(folder: pathlib.Path) -> List[Dict[str, str | int]]:
    return [
        r for f in folder.iterdir() if LOG_STEM.match(f.stem)
        for r in [analyze_file(f)] if r
    ]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Scan log_*.txt files and report protocol successes / failures"
    )
    # ap.add_argument("folder", help="Folder containing the log files")
    # ap.add_argument("--csv", help="Optional: write the summary table to this CSV")
    # args = ap.parse_args()

    folder = Path(r"C:\Users\engmo\Desktop\r1_project\gaskin_logs\june12")

    folder = pathlib.Path(folder).expanduser()
    summaries = analyze_folder(folder)

    # ---------- NEW: overall totals ----------
    total_success = sum(r["success"] for r in summaries)
    total_failure = sum(r["failure"] for r in summaries)
    # -----------------------------------------

    # pretty-print per-file JSON
    # print(json.dumps(summaries, indent=2))

    # NEW: print the grand totals
    print(f"\nOverall summary: {total_success} successes, {total_failure} failures")

    if summaries:
        with open("gaskin_june_12_result.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Result was saved successfully")