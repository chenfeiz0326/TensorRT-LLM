#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate a pre-merge HTML report with inline SVG performance charts.

Queries OpenSearch directly for pre-merge data (by user + job_id) and
post-merge history, then generates an HTML report visualizing key
throughput metrics with history, new data, baseline, and threshold lines
for regression comparison.

Pipeline overview
=================
This script is invoked standalone (not inside the TRT-LLM test container),
so it cannot import test-side modules.  It only depends on:
  - ``perf_utils``   (sibling file — constants, OpenSearch helpers, SVG helpers)
  - ``open_search_db`` (parent dir — low-level OpenSearch HTTP client)

Execution steps (see ``main()``):

1. **Parse CLI args**
   ``--user`` and ``--job-id`` identify the merge-request run.
   ``--html`` is the output path (default ``perf_sanity_report.html``).

2. **Query pre-merge data** — ``query_pre_merge_data(user, job_id)``
   Sends an OpenSearch query filtered by:
     ``b_is_valid=True, b_is_post_merge=False,
      s_trigger_mr_user=<user>, s_job_id=<job_id>``
   Returns a flat list of perf-data dicts uploaded by the test run.

3. **Query post-merge history** — ``get_pre_merge_history_data(new_data_list)``
   Calls ``perf_utils.get_history_data()`` with
     ``b_is_post_merge=True, s_branch=main``
   to fetch the last QUERY_LOOKBACK_DAYS of post-merge results (both
   regular history and baseline entries).  The result is grouped by
   ``(s_test_case_name, s_gpu_type)`` and filtered to only the keys
   present in the pre-merge data.

4. **Generate HTML report** — ``generate_pre_merge_html(...)``
   For each ``(test_case, gpu_type)`` group:
     a. Extract ``s_stage_name`` and ``s_test_list`` metadata from the
        pre-merge entries (used for the "Stage Name" / "Pytest Command
        to Repro" footer below each test case).
     b. For each of the 4 ``CHART_METRICS`` (seq_throughput,
        token_throughput, total_token_throughput, user_throughput):
          - Compute history points, baseline value, threshold value, and
            latest pre-merge value.
          - Render an inline SVG chart via ``_generate_pre_merge_chart()``.
            Chart legend:
              * Blue solid line   — historical post-merge data
              * Red dashed line   — baseline value
              * Yellow dashed line — threshold value
              * Purple dashed line — latest pre-merge performance
     c. Append the section HTML (collapsible ``<details>`` block) with
        the 4 charts and footer metadata.
   Finally, wrap all sections in a full HTML document and write to
   ``output_file``.
"""

import argparse
import os
import sys
from html import escape as escape_html

# Set OPEN_SEARCH_DB_BASE_URL before importing perf_utils, because
# open_search_db captures the env var at module-import time.
if not os.environ.get("OPEN_SEARCH_DB_BASE_URL"):
    os.environ["OPEN_SEARCH_DB_BASE_URL"] = "http://gpuwa.nvidia.com"

from perf_utils import (
    CHART_METRICS,
    MAX_QUERY_SIZE,
    METRIC_LABELS,
    PERF_SANITY_PROJECT_NAME,
    QUERY_LOOKBACK_DAYS,
    _MARGIN,
    _PLOT_H,
    _PLOT_W,
    _SVG_HEIGHT,
    _SVG_WIDTH,
    _extract_points,
    _get_threshold_for_metric,
    _ts_to_date,
    get_history_data,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from open_search_db import OpenSearchDB

# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def query_pre_merge_data(user, job_id):
    """Query OpenSearch for pre-merge perf data matching the given user and job_id.

    Returns:
        list of data dicts, or empty list on failure.
    """
    must_clauses = [
        {"term": {"b_is_valid": True}},
        {"term": {"b_is_post_merge": False}},
        {"term": {"s_trigger_mr_user": user}},
        {"term": {"s_job_id": job_id}},
    ]

    data_list = OpenSearchDB.queryPerfDataFromOpenSearchDB(
        PERF_SANITY_PROJECT_NAME, must_clauses, size=MAX_QUERY_SIZE
    )

    if data_list is None:
        print("Warning: Failed to query pre-merge data from OpenSearch")
        return []

    return data_list


# ---------------------------------------------------------------------------
# History data query
# ---------------------------------------------------------------------------


def get_pre_merge_history_data(new_data_list):
    """Query OpenSearch for post-merge history data matching test cases in *new_data_list*.

    Uses :func:`perf_utils.get_history_data` to fetch post-merge history
    (both baseline and non-baseline), then filters to only the
    (s_test_case_name, s_gpu_type) pairs present in *new_data_list*.

    Returns:
        dict mapping (test_case, gpu_type) -> {
            "history_data": [...],
            "baseline_data": [...],
        }
        or empty dict on failure / no matches.
    """
    if not new_data_list:
        return {}

    # Determine which test case keys are present in new data
    needed_keys = set()
    for nd in new_data_list:
        key = (nd.get("s_test_case_name", ""), nd.get("s_gpu_type", ""))
        needed_keys.add(key)

    grouped = get_history_data(
        extra_must_clauses=[
            {"term": {"b_is_post_merge": True}},
            {"term": {"s_branch": "main"}},
        ]
    )

    if grouped is None:
        print("Warning: Failed to query history data from OpenSearch")
        return {}

    # Filter to only the test cases we have new data for
    filtered = {}
    for key, bucket in grouped.items():
        if key in needed_keys:
            filtered[key] = bucket

    return filtered


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def _extract_simple_points(data_list, metric):
    """Extract (datetime, float_value) pairs from a list of data dicts."""
    points = []
    for d in data_list:
        ts = d.get("ts_created") or d.get("@timestamp")
        val = d.get(metric)
        if ts is not None and val is not None:
            try:
                points.append((_ts_to_date(ts), float(val)))
            except (ValueError, TypeError):
                pass
    points.sort(key=lambda p: p[0])
    return points


# Colors for the pre-merge chart
_COLOR_HISTORY = "#4285f4"       # blue — historical post-merge data
_COLOR_BASELINE = "#d93025"      # red — baseline value
_COLOR_THRESHOLD = "#daa520"     # yellow — threshold value
_COLOR_PRE_MERGE = "#7b2d8e"     # purple — latest pre-merge performance


def _generate_pre_merge_chart(
    history_points,
    metric,
    label,
    baseline_value=None,
    threshold_line_value=None,
    pre_merge_value=None,
):
    """Return an SVG string for a single pre-merge metric chart.

    Renders:
      - Blue line graph: historical post-merge data over time
      - Red dashed line: baseline value
      - Yellow dashed line: threshold value
      - Purple dashed line: latest pre-merge performance value
    """
    all_values = [v for _, v, *_ in history_points if v is not None]
    if baseline_value is not None:
        all_values.append(baseline_value)
    if threshold_line_value is not None:
        all_values.append(threshold_line_value)
    if pre_merge_value is not None:
        all_values.append(pre_merge_value)

    if not history_points and baseline_value is None and pre_merge_value is None:
        return f'<div style="color:#888;padding:10px;">No data for {escape_html(label)}</div>'
    if not all_values:
        return (
            f'<div style="color:#888;padding:10px;">No numeric data for {escape_html(label)}</div>'
        )

    min_val = min(all_values)
    max_val = max(all_values)
    val_range = max_val - min_val if max_val != min_val else 1.0
    min_val -= val_range * 0.05
    max_val += val_range * 0.05
    val_range = max_val - min_val

    dates = [d for d, *_ in history_points]
    if not dates:
        return (
            f'<div style="color:#888;padding:10px;">No data points for {escape_html(label)}</div>'
        )

    min_ts = min(dates).timestamp()
    max_ts = max(dates).timestamp()
    ts_range = max_ts - min_ts if max_ts != min_ts else 1.0

    def _x(dt):
        return _MARGIN["left"] + (dt.timestamp() - min_ts) / ts_range * _PLOT_W

    def _y(v):
        return _MARGIN["top"] + _PLOT_H - (v - min_val) / val_range * _PLOT_H

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_WIDTH}" '
        f'height="{_SVG_HEIGHT}" style="background:#fff;border:1px solid #ddd;'
        f'border-radius:4px;margin:5px 0;">'
    ]

    # Grid lines (Y axis, 5 ticks)
    for i in range(6):
        v = min_val + val_range * i / 5
        y = _y(v)
        svg.append(
            f'<line x1="{_MARGIN["left"] - 4}" y1="{y:.1f}" '
            f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{y:.1f}" '
            f'stroke="#eee" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{_MARGIN["left"] - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end" font-size="10" fill="#666">{v:.1f}</text>'
        )

    # Axes
    svg.append(
        f'<line x1="{_MARGIN["left"]}" y1="{_MARGIN["top"]}" '
        f'x2="{_MARGIN["left"]}" y2="{_MARGIN["top"] + _PLOT_H}" '
        f'stroke="#ccc" stroke-width="1"/>'
    )
    svg.append(
        f'<line x1="{_MARGIN["left"]}" y1="{_MARGIN["top"] + _PLOT_H}" '
        f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{_MARGIN["top"] + _PLOT_H}" '
        f'stroke="#ccc" stroke-width="1"/>'
    )

    # X-axis date labels
    unique_dates = sorted(set(dates))
    n_labels = min(6, len(unique_dates))
    if len(unique_dates) >= n_labels:
        label_dates = unique_dates[:: max(1, len(unique_dates) // n_labels)][:n_labels]
    else:
        label_dates = unique_dates
    for dt in label_dates:
        x = _x(dt)
        y_base = _MARGIN["top"] + _PLOT_H
        svg.append(
            f'<text x="{x:.1f}" y="{y_base + 18}" text-anchor="middle" '
            f'font-size="10" fill="#666">{dt.strftime("%m/%d")}</text>'
        )

    # Title
    title_text = escape_html(label)
    svg.append(
        f'<text x="{_SVG_WIDTH / 2}" y="{_MARGIN["top"] - 8}" '
        f'text-anchor="middle" font-size="12" font-weight="bold" '
        f'fill="#333">{title_text}</text>'
    )

    # Baseline horizontal line (red dashed)
    if baseline_value is not None:
        by = _y(baseline_value)
        svg.append(
            f'<line x1="{_MARGIN["left"]}" y1="{by:.1f}" '
            f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{by:.1f}" '
            f'stroke="{_COLOR_BASELINE}" stroke-width="1.5" stroke-dasharray="6,3"/>'
        )

    # Threshold horizontal line (yellow dashed)
    if threshold_line_value is not None:
        ty = _y(threshold_line_value)
        svg.append(
            f'<line x1="{_MARGIN["left"]}" y1="{ty:.1f}" '
            f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{ty:.1f}" '
            f'stroke="{_COLOR_THRESHOLD}" stroke-width="1.5" stroke-dasharray="4,4"/>'
        )

    # Pre-merge value horizontal line (purple dashed)
    if pre_merge_value is not None:
        py = _y(pre_merge_value)
        svg.append(
            f'<line x1="{_MARGIN["left"]}" y1="{py:.1f}" '
            f'x2="{_MARGIN["left"] + _PLOT_W}" y2="{py:.1f}" '
            f'stroke="{_COLOR_PRE_MERGE}" stroke-width="1.5" stroke-dasharray="5,3"/>'
        )

    # History line + dots (blue)
    sorted_hist = sorted(
        [(d, v, *rest) for d, v, *rest in history_points if v is not None],
        key=lambda p: p[0],
    )
    if len(sorted_hist) > 1:
        path_d = " ".join(
            f"{'M' if i == 0 else 'L'}{_x(d):.1f},{_y(v):.1f}"
            for i, (d, v, *_) in enumerate(sorted_hist)
        )
        svg.append(
            f'<path d="{path_d}" fill="none" stroke="{_COLOR_HISTORY}" stroke-width="2"/>'
        )
    for d, v, *_ in sorted_hist:
        svg.append(
            f'<circle cx="{_x(d):.1f}" cy="{_y(v):.1f}" r="3" fill="{_COLOR_HISTORY}">'
            f"<title>{d.strftime('%Y-%m-%d %H:%M')}  {v:.2f}</title></circle>"
        )

    # Legend
    legend_y = _MARGIN["top"] + _PLOT_H + 35
    legend_x = _MARGIN["left"] + 10

    # History legend
    svg.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4" fill="{_COLOR_HISTORY}"/>')
    svg.append(
        f'<text x="{legend_x + 8}" y="{legend_y + 4}" font-size="10" fill="#666">History</text>'
    )
    legend_x += 65

    # Baseline legend
    if baseline_value is not None:
        svg.append(
            f'<line x1="{legend_x}" y1="{legend_y}" '
            f'x2="{legend_x + 20}" y2="{legend_y}" '
            f'stroke="{_COLOR_BASELINE}" stroke-width="1.5" stroke-dasharray="6,3"/>'
        )
        svg.append(
            f'<text x="{legend_x + 25}" y="{legend_y + 4}" '
            f'font-size="10" fill="#666">Baseline ({baseline_value:.2f})</text>'
        )
        legend_x += 145

    # Threshold legend
    if threshold_line_value is not None:
        svg.append(
            f'<line x1="{legend_x}" y1="{legend_y}" '
            f'x2="{legend_x + 20}" y2="{legend_y}" '
            f'stroke="{_COLOR_THRESHOLD}" stroke-width="1.5" stroke-dasharray="4,4"/>'
        )
        svg.append(
            f'<text x="{legend_x + 25}" y="{legend_y + 4}" '
            f'font-size="10" fill="#666">Threshold ({threshold_line_value:.2f})</text>'
        )
        legend_x += 145

    # Pre-merge legend
    if pre_merge_value is not None:
        svg.append(
            f'<line x1="{legend_x}" y1="{legend_y}" '
            f'x2="{legend_x + 20}" y2="{legend_y}" '
            f'stroke="{_COLOR_PRE_MERGE}" stroke-width="1.5" stroke-dasharray="5,3"/>'
        )
        svg.append(
            f'<text x="{legend_x + 25}" y="{legend_y + 4}" '
            f'font-size="10" fill="#666">Pre-merge ({pre_merge_value:.2f})</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def generate_pre_merge_html(new_data_list, history_grouped, output_file):
    """Generate HTML report visualizing new data against history + baseline.

    For each (test_case, gpu_type) present in *new_data_list*, renders 4
    charts (one per key metric) showing history line, new data points,
    baseline line, and threshold line for regression comparison.

    Each test case section includes s_stage_name and s_test_list metadata
    from the pre-merge data.
    """
    # Group new data by (test_case, gpu_type)
    new_groups = {}
    for nd in new_data_list:
        key = (nd.get("s_test_case_name", ""), nd.get("s_gpu_type", ""))
        new_groups.setdefault(key, []).append(nd)

    sections_html = []
    for (test_case, gpu_type), new_data_entries in sorted(new_groups.items()):
        bucket = history_grouped.get((test_case, gpu_type), {})
        history_data = bucket.get("history_data", [])
        baseline_data_list = bucket.get("baseline_data", [])

        # Extract stage_name and test_list from pre-merge data
        stage_name = ""
        test_list = ""
        for nd in new_data_entries:
            if nd.get("s_stage_name"):
                stage_name = nd["s_stage_name"]
            if nd.get("s_test_list"):
                test_list = nd["s_test_list"]
            if stage_name and test_list:
                break

        charts = []
        for metric in CHART_METRICS:
            label = METRIC_LABELS.get(metric, metric)

            # History points (blue line) — use 3-tuple version from perf_utils
            hist_pts = _extract_points(history_data, metric)

            # Latest pre-merge performance value (purple dashed line)
            pre_merge_value = None
            pre_merge_pts = _extract_simple_points(new_data_entries, metric)
            if pre_merge_pts:
                # Use the latest pre-merge data point
                pre_merge_value = pre_merge_pts[-1][1]

            # Baseline value from the latest baseline entry
            baseline_value = None
            if baseline_data_list:
                latest_bl = baseline_data_list[-1]
                bl_val = latest_bl.get(metric)
                if bl_val is not None:
                    baseline_value = float(bl_val)

            # Threshold line value
            threshold_line_value = None
            if baseline_value is not None:
                threshold = _get_threshold_for_metric(baseline_data_list, metric)
                threshold_line_value = baseline_value * (1 - threshold)

            charts.append(
                _generate_pre_merge_chart(
                    hist_pts,
                    metric,
                    label,
                    baseline_value=baseline_value,
                    threshold_line_value=threshold_line_value,
                    pre_merge_value=pre_merge_value,
                )
            )

        header = escape_html(f"{test_case}  [{gpu_type}]")
        repro_cmd = f"perf/test_perf_sanity.py::test_e2e[{test_list}]" if test_list else ""
        meta_html = f'<div class="meta-info">'
        meta_html += f'<div><span class="meta-label">Stage Name:</span> {escape_html(stage_name)}</div>'
        meta_html += f'<div><span class="meta-label">Pytest Command to Repro:</span> <code>{escape_html(repro_cmd)}</code></div>'
        meta_html += '</div>'

        section = f"""
        <details class="test-section" open>
            <summary><strong>{header}</strong></summary>
            <div class="charts-grid">
                {"".join(charts)}
            </div>
            {meta_html}
        </details>
        """
        sections_html.append(section)

    total_new = len(new_data_list)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Perf Sanity Pre-Merge Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }}
        h2 {{ color: #333; }}
        .test-section {{
            margin-bottom: 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: #fff;
            padding: 10px 15px;
        }}
        .test-section summary {{
            cursor: pointer;
            font-size: 14px;
            padding: 6px 0;
        }}
        .charts-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        .summary-info {{ color: #666; margin-bottom: 15px; }}
        .meta-info {{ color: #555; font-size: 13px; margin-top: 10px; }}
        .meta-info div {{ margin-bottom: 2px; }}
        .meta-label {{ font-weight: bold; }}
        .meta-info code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}
    </style>
</head>
<body>
    <h2>Perf Sanity Pre-Merge Results</h2>
    <p class="summary-info">{len(new_groups)} test case(s) &middot; {total_new} new data point(s)</p>
    {"".join(sections_html)}
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated pre-merge perf report with {len(new_groups)} test cases: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate a pre-merge HTML report with historical "
        "performance charts, baseline, and threshold lines. "
        "Queries OpenSearch directly by user and job ID."
    )
    parser.add_argument(
        "--user",
        type=str,
        required=True,
        help="Filter pre-merge data by s_trigger_mr_user",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="Filter pre-merge data by s_job_id",
    )
    parser.add_argument(
        "--html",
        type=str,
        default="perf_sanity_report.html",
        help="Output HTML file path (default: perf_sanity_report.html)",
    )
    args = parser.parse_args()

    new_data_list = query_pre_merge_data(args.user, args.job_id)
    if not new_data_list:
        print("No pre-merge perf data found for the given user and job ID.")
        # Still generate an empty report
        generate_pre_merge_html([], {}, args.html)
        return

    print(f"Found {len(new_data_list)} pre-merge data entries")
    history_grouped = get_pre_merge_history_data(new_data_list)
    generate_pre_merge_html(new_data_list, history_grouped, args.html)


if __name__ == "__main__":
    main()
