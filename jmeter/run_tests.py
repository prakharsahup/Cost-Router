"""Automated JMeter Performance Test Runner and Results Aggregator.

1. Starts CostRouter API server in background.
2. Executes Apache JMeter test plans in non-GUI mode.
3. Generates HTML dashboard reports.
4. Parses raw JTL/CSV data to compute KPIs:
   - Request counts, error rate
   - Throughput (req/sec)
   - Latencies: Mean, Min, Max, p50, p90, p95, p99
5. Shuts down background server.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Any


# Set JMETER_HOME to your JMeter install (the folder containing bin/).
JMETER_HOME = Path(os.environ.get("JMETER_HOME", r"C:\apache-jmeter-5.6.3"))
JMETER_BAT = JMETER_HOME / "bin" / ("jmeter.bat" if os.name == "nt" else "jmeter")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
JMETER_DIR = PROJECT_ROOT / "jmeter"
RESULTS_DIR = JMETER_DIR / "results"
REPORTS_DIR = JMETER_DIR / "reports"


def check_jmeter():
    if not JMETER_BAT.exists():
        raise FileNotFoundError(f"JMeter executable not found at {JMETER_BAT} — set JMETER_HOME")
    print(f"[OK] JMeter binary confirmed at: {JMETER_BAT}")


def clean_and_prepare_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server(port: int = 8000, timeout: float = 10.0) -> bool:
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def run_jmeter_plan(plan_path: Path, jtl_path: Path, report_dir: Path):
    if jtl_path.exists():
        jtl_path.unlink()
    if report_dir.exists():
        import shutil
        shutil.rmtree(report_dir)

    cmd = [
        str(JMETER_BAT),
        "-n",
        "-t", str(plan_path),
        "-l", str(jtl_path),
        "-e",
        "-o", str(report_dir),
    ]

    print(f"\n[RUN] Executing JMeter: {' '.join(cmd)}")
    start_t = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    elapsed = time.time() - start_t
    print(f"[DONE] Execution finished in {elapsed:.2f}s (Exit code: {res.returncode})")
    if res.returncode != 0:
        print("[STDERR]:", res.stderr)
    return res.returncode == 0


def parse_jtl(jtl_path: Path) -> List[Dict[str, Any]]:
    if not jtl_path.exists():
        return []

    rows = []
    with open(jtl_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "timestamp": int(row.get("timeStamp", 0)),
                    "elapsed": float(row.get("elapsed", 0)),
                    "label": row.get("label", ""),
                    "responseCode": row.get("responseCode", ""),
                    "success": row.get("success", "true").lower() == "true",
                    "bytes": int(row.get("bytes", 0)),
                    "sentBytes": int(row.get("sentBytes", 0)),
                })
            except Exception:
                continue
    return rows


def compute_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {}

    labels = sorted(list(set(s["label"] for s in samples)))
    all_res = {}

    def stats_for_group(group: List[Dict[str, Any]]):
        count = len(group)
        errors = sum(1 for s in group if not s["success"])
        latencies = sorted(s["elapsed"] for s in group)
        total_time_ms = max(s["timestamp"] + s["elapsed"] for s in group) - min(s["timestamp"] for s in group)
        total_time_sec = max(0.001, total_time_ms / 1000.0)

        def pct(p):
            k = int(round((len(latencies) - 1) * p / 100.0))
            return latencies[k]

        return {
            "samples": count,
            "errors": errors,
            "error_pct": round(errors / count * 100.0, 2),
            "throughput": round(count / total_time_sec, 2),
            "mean_ms": round(sum(latencies) / count, 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "p50_ms": round(pct(50), 2),
            "p90_ms": round(pct(90), 2),
            "p95_ms": round(pct(95), 2),
            "p99_ms": round(pct(99), 2),
        }

    for label in labels:
        grp = [s for s in samples if s["label"] == label]
        all_res[label] = stats_for_group(grp)

    all_res["TOTAL"] = stats_for_group(samples)
    return all_res


def print_table(title: str, metrics: Dict[str, Any]):
    print(f"\n{'='*95}")
    print(f" {title.center(93)}")
    print(f"{'='*95}")
    header = f"{'Sampler / Label':<30} | {'Count':>6} | {'Err%':>6} | {'Throughput':>10} | {'Avg (ms)':>8} | {'p90 (ms)':>8} | {'p95 (ms)':>8} | {'p99 (ms)':>8}"
    print(header)
    print("-" * 95)
    for label, stat in metrics.items():
        if label == "TOTAL":
            print("-" * 95)
        print(
            f"{label:<30} | {stat['samples']:>6} | {stat['error_pct']:>5.1f}% | "
            f"{stat['throughput']:>8.1f}/s | {stat['mean_ms']:>8.1f} | "
            f"{stat['p90_ms']:>8.1f} | {stat['p95_ms']:>8.1f} | {stat['p99_ms']:>8.1f}"
        )
    print(f"{'='*95}\n")


def main():
    check_jmeter()
    clean_and_prepare_dirs()

    # 1. Start Server
    server_process = subprocess.Popen(
        [sys.executable, "-m", "costrouter.api_server", "8000"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("[INIT] Started CostRouter API Server in background...")

    try:
        ready = wait_for_server(8000, 10.0)
        if not ready:
            raise RuntimeError("CostRouter API server failed to become ready.")
        print("[OK] CostRouter API Server is responding on http://127.0.0.1:8000/health")

        # 2. Test Plan 1: CostRouter
        cr_plan = JMETER_DIR / "costrouter_load_test.jmx"
        cr_jtl = RESULTS_DIR / "costrouter_results.jtl"
        cr_report = REPORTS_DIR / "costrouter_report"
        run_jmeter_plan(cr_plan, cr_jtl, cr_report)

        cr_samples = parse_jtl(cr_jtl)
        cr_metrics = compute_metrics(cr_samples)
        print_table("CostRouter Load Test Results (Local API)", cr_metrics)

        # 3. Test Plan 2: Web App
        web_plan = JMETER_DIR / "web_app_load_test.jmx"
        web_jtl = RESULTS_DIR / "webapp_results.jtl"
        web_report = REPORTS_DIR / "webapp_report"
        run_jmeter_plan(web_plan, web_jtl, web_report)

        web_samples = parse_jtl(web_jtl)
        web_metrics = compute_metrics(web_samples)
        print_table("Web App Load Test Results (the-internet.herokuapp.com)", web_metrics)

        return cr_metrics, web_metrics

    finally:
        print("[CLEANUP] Stopping CostRouter API Server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("[CLEANUP] Server stopped.")


if __name__ == "__main__":
    main()
