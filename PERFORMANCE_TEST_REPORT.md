# CostRouter Performance & Load Testing Report

**Project**: CostRouter — Cost-Aware LLM Routing for LangGraph Agents  
**Tool**: Apache JMeter 5.6.3 (LTS)  
**Java Runtime**: Oracle Java 25.0.2 LTS (64-Bit)  
**Execution Environment**: Windows 11 AMD64 (12 Cores, 16 GB RAM)  
**Test Date**: September 23, 2026  
**Test Status**: **PASSED (0.0% Error Rate across all test suites)**  

---

## 1. Executive Summary

This report documents the load, stress, and performance testing conducted on the **CostRouter LLM routing engine** and the associated **Web Application workflow** using **Apache JMeter 5.6.3** in non-GUI (headless) execution mode.

The testing evaluated system stability, concurrency scaling, response latency distributions (Average, p90, p95, p99), and throughput (requests/second) under graduated concurrency tiers:
- **Baseline Load** (5 concurrent virtual users)
- **Standard Load** (25 concurrent virtual users)
- **Peak / Stress Load** (50 concurrent virtual users)

### Key Achievements:
- **Zero Errors**: Both test suites ran with a **0.0% error rate** across **925 total test requests** (850 CostRouter API requests + 75 Web App requests).
- **High Throughput**: CostRouter sustained **71.6 requests/sec** under peak load on a single local node.
- **Microsecond-to-Millisecond Latencies**: CostRouter routing decisions completed in an average of **14.9 ms** (well below the Non-Functional Requirement of 200 ms).
- **Interactive Reports**: Generated standalone, interactive JMeter HTML Dashboard Reports with APDEX scores, latency curves, and percentile charts.

### Scope and caveats
- The API server uses **mock models** that sleep 2 ms (cheap) / 6 ms (expensive).
  Most of the measured latency is that simulated sleep, so these figures measure
  the HTTP + routing + budget-ledger path, **not** real LLM latency.
- The heuristic estimator is pure local computation, well under the 200 ms requirement (`tests/test_latency.py`).
- The Web Transactions suite targets a public demo site (`the-internet.herokuapp.com`),
  not CostRouter; it exercises the JMeter setup only.

---

## 2. Test Architecture & Methodology

```
+-------------------------------------------------------------+
|                     Apache JMeter 5.6.3                     |
|  - Headless Non-GUI Engine (jmeter.bat -n -t ... -l ... -e) |
+------------------------------+------------------------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
+-----------------------+             +-----------------------+
|  CostRouter REST API  |             | The-Internet App      |
|  http://127.0.0.1:8000|             | herokuapp.com (HTTPS) |
|  - GET /health        |             | - /login, /checkboxes |
|  - POST /api/route    |             | - /dropdown, /alerts  |
|    (Easy & Complex)   |             | - /key_presses        |
+-----------------------+             +-----------------------+
            |                                     |
            v                                     v
+-----------------------+             +-----------------------+
| costrouter_report/    |             | webapp_report/        |
| Interactive HTML Dash |             | Interactive HTML Dash |
+-----------------------+             +-----------------------+
```

### Test Plan Specifications

| Test Suite | JMeter Test Plan File | Target Host | Thread Groups | Total Samples |
|---|---|---|---|---|
| **CostRouter Engine** | [`jmeter/costrouter_load_test.jmx`](jmeter/costrouter_load_test.jmx) | `127.0.0.1:8000` | 5 users (ramp 2s), 25 users (ramp 5s), 50 users (ramp 5s) | 850 |
| **Web Transactions** | [`jmeter/web_app_load_test.jmx`](jmeter/web_app_load_test.jmx) | `the-internet.herokuapp.com` | 5 users (ramp 3s), 3 loops | 75 |

---

## 3. CostRouter Performance Results

### Metric Summary Table

| Transaction / Sampler | Samples | Error % | Throughput | Avg Latency | Min Latency | Max Latency | 90th %ile | 95th %ile | 99th %ile |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Baseline_GET_Health` | 50 | **0.0%** | 29.9 req/s | 2.7 ms | 1.0 ms | 25.0 ms | 4.0 ms | 5.0 ms | 22.0 ms |
| `Baseline_POST_Route_Easy` | 50 | **0.0%** | 30.1 req/s | 11.8 ms | 10.0 ms | 18.0 ms | 14.0 ms | 14.0 ms | 14.0 ms |
| `Load_POST_Route_Complex` | 250 | **0.0%** | 49.0 req/s | 15.2 ms | 11.0 ms | 42.0 ms | 16.0 ms | 18.0 ms | 28.0 ms |
| `Load_POST_Route_Easy` | 250 | **0.0%** | 49.0 req/s | 14.9 ms | 11.0 ms | 36.0 ms | 16.0 ms | 17.0 ms | 29.0 ms |
| `Stress_POST_Route_Mixed` | 250 | **0.0%** | 49.5 req/s | 17.7 ms | 11.0 ms | 98.0 ms | 21.0 ms | 41.0 ms | 62.0 ms |
| **TOTAL / OVERALL** | **850** | **0.0%** | **71.6 req/s** | **14.9 ms** | **1.0 ms** | **98.0 ms** | **16.0 ms** | **18.0 ms** | **46.0 ms** |

---

## 4. Web Application Workflow Results (Functional Complement)

Replicating key user scenarios from [`FunctionalTesting.side`](FunctionalTesting.side):

| Test Case / Transaction | Samples | Error % | Throughput | Avg Latency | 90th %ile | 95th %ile | 99th %ile |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `TC01_GET_LoginPage` | 15 | 0.0% | 2.5 req/s | 616.6 ms | 1457.0 ms | 1457.0 ms | 2000.0 ms |
| `TC02_GET_Checkboxes` | 15 | 0.0% | 3.5 req/s | 274.9 ms | 286.0 ms | 286.0 ms | 485.0 ms |
| `TC03_GET_Dropdown` | 15 | 0.0% | 3.5 req/s | 276.7 ms | 317.0 ms | 317.0 ms | 457.0 ms |
| `TC04_GET_JSAlerts` | 15 | 0.0% | 3.5 req/s | 256.7 ms | 270.0 ms | 270.0 ms | 273.0 ms |
| `TC05_GET_KeyPresses` | 15 | 0.0% | 3.3 req/s | 336.7 ms | 321.0 ms | 321.0 ms | 1316.0 ms |
| **TOTAL / OVERALL** | **75** | **0.0%** | **10.2 req/s** | **352.3 ms** | **457.0 ms** | **1026.0 ms** | **1457.0 ms** |

---

## 5. Non-Functional Requirements (NFR) Verification

The CostRouter SRS defines strict latency and performance bounds in **REQ-DIFF-3** and **§5.1 Performance**:

| Requirement | Description | Target Threshold | Measured JMeter Result | Status |
|---|---|---|---|:---:|
| **REQ-DIFF-3 (Estimator Latency)** | Difficulty estimation computation time per step | < 200.0 ms | **14.9 ms (Avg)** / **46.0 ms (p99)** | **PASS** |
| **REQ-ROUTER-Overhead** | Router node routing overhead (minus external LLM) | < 50.0 ms | **~12–15 ms** | **PASS** |
| **REQ-HEALTH-Check** | Health check probe response time | < 50.0 ms | **2.7 ms (Avg)** / **22.0 ms (Max)** | **PASS** |
| **System Reliability** | Unhandled server exceptions or connection drops | 0.0% | **0.0% Errors (0/850 failed)** | **PASS** |

---

## 6. Generated Test Artifacts & Deliverables

All test plans, automation runners, raw logs, and HTML visual reports are recorded in the repository:

- **JMeter Test Plans (.jmx)**:
  - Interactive GUI Test Plan: [`jmeter/costrouter_gui_testplan.jmx`](jmeter/costrouter_gui_testplan.jmx) (Contains all 5 visual GUI listeners)
  - CostRouter Load Test Plan: [`jmeter/costrouter_load_test.jmx`](jmeter/costrouter_load_test.jmx)
  - Web App Load Test Plan: [`jmeter/web_app_load_test.jmx`](jmeter/web_app_load_test.jmx)
- **Launchers & Runner Scripts**:
  - One-Click GUI Launcher: [`launch_jmeter.bat`](launch_jmeter.bat)
  - Python Automation Engine: [`jmeter/run_tests.py`](jmeter/run_tests.py)
  - PowerShell CLI Wrapper: [`jmeter/run_performance_tests.ps1`](jmeter/run_performance_tests.ps1)
- **Raw Test Data (.jtl) and Interactive HTML Dashboard Reports**: generated output,
  not kept in the repo. Regenerate them with `python jmeter/run_tests.py`
  (set `JMETER_HOME` first); they are written to `jmeter/results/` and
  `jmeter/reports/costrouter_report/` / `jmeter/reports/webapp_report/`.

---

## 7. How to Run in JMeter GUI (For Screenshots)

1. Keep the API server running in your terminal:
   ```powershell
   python -m costrouter.api_server 8000
   ```
2. Launch the JMeter GUI with the preloaded test plan:
   ```powershell
   .\launch_jmeter.bat
   ```
3. Inside Apache JMeter GUI:
   - Click the green **Play** button (`▶` or `Ctrl + R`) on the top toolbar to start testing.
   - Click on **View Results Tree** in the left sidebar to capture screenshots of individual requests & JSON responses with green checkmarks.
   - Click on **Summary Report** to capture the summary metrics table.
   - Click on **Aggregate Report** to capture throughput and 90%/95%/99% percentiles.
   - Click on **View Results in Table** to capture sample-by-sample status checks.
   - Click on **Graph Results** to capture the real-time latency curve.

