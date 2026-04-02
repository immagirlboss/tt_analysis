# TikTok LIVE Analysis

This project generates a synthetic TikTok LIVE moderation dataset, runs a small SQL-based analysis on the outputs, and builds a dashboard that summarizes the findings.

## What it does

- Creates synthetic moderation event data and CCV time-series data
- Analyzes the data with pandas + SQLite-style queries
- Produces summary tables for the dashboard
- Builds a four-panel visualization of the main findings

## Files

- `data_gen.py` - generates `moderation_events.csv` and `ccv_timeseries.csv`
- `analysis.py` - computes the summary findings and writes `summary_*.csv`
- `viz.py` - builds `engagement_dropoff_dashboard.png`

## Requirements

- pandas
- numpy
- matplotlib

## How to run

Run the scripts in this order from the project root:

```bash
python data_gen.py
python analysis.py
python viz.py
```

## Outputs

After a successful run, you should see these generated files:

- `moderation_events.csv`
- `ccv_timeseries.csv`
- `summary_intervention.csv`
- `summary_policy.csv`
- `summary_detection.csv`
- `summary_impact.csv`
- `engagement_dropoff_dashboard.png`

## Notes

- The datasets are synthetic, but they are calibrated to public TikTok transparency-report ratios.
- The analysis is descriptive and should be treated as an internal decision-support report, not causal proof.
- The dashboard is designed to surface operational priorities around retention, creator experience, and moderation workflow quality.
