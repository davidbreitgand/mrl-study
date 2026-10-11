# Harness source

The Python source, `Dockerfile`, `pyproject.toml`, `uv.lock`, `src/`,
and `tests/` in this directory make up the harness image that the
`make images` target builds and pushes. The harness runs as a
Kubernetes Job (`stack/harness/job.yaml.tmpl`) once per cell; each
invocation drives an open-loop arrival process against the deployed
Hub EPP, journals every completion, and computes per-endpoint
attribution.

## Build

```bash
export OWNER=your-registry-namespace
scripts/build-harness-image.sh
```

Writes `.harness-image` next to the study root with the image
reference. `scripts/apply-harness-job.sh` reads that file to
render the Job manifest with the right tag.

## Open-loop arrival process

`DatasetRunner` dispatches requests as an open-loop process. Two
arrival distributions are supported and switched by
`config/arrival-regimes.yaml`:

- **Poisson**: inter-arrival gaps drawn from `Exponential(λ)`.
- **Weibull-<shape>**: inter-arrival gaps drawn from
  `Weibull(shape, λ)` with configurable shape (0.5 and 0.3 in the
  head-to-head study).

`λ` is set by the cell's calibrated `study_lambda` (from
`local_results/<dataset>/calibration.json`) or by an operator-supplied
override.

At end of dataset the driver stops spawning new rows and awaits every
in-flight task before returning (drain). No wall-clock cutoff other
than the Job's `activeDeadlineSeconds`.

Journal records carry two arrival-process fields under `extra_fields`:

- `arrival_ts_seconds` — seconds since `run()` start when the arrival
  fired.
- `in_flight_at_arrival` — number of outstanding row tasks at the
  moment this arrival was spawned. Used by `probe_stage_analyze.py`
  and the aggregators to compute `ρ_honest`.

## Per-endpoint attribution

The harness reads the response header
`x-conformance-test-served-endpoint` on every completion to know
which spoke served the request. That endpoint name lands in
`extra_fields.served_endpoint` on every journal row and drives:

- Arm attribution (spoke-a vs spoke-b share).
- Per-endpoint cost attribution via `clusters.yaml`.

## Attribution formula

Per record, computed at record-write time in `cost.py`:

    gpu_hour_cost = elapsed_seconds × tp × rate / 3600 / 8

Where `tp` and `rate` come from `clusters.yaml`'s `llm-d.ai/tp-size`
and `llm-d.ai/gpu-hourly-rate-usd` labels for the served endpoint,
loaded at harness startup from
`--clusters-yaml PATH` (default `/etc/harness/clusters.yaml`, mounted
via ConfigMap by the Job manifest). If the response header is missing
or the endpoint name is not in the clusters table, cost is recorded
as 0.0 and the analysis pass surfaces this as "unattributed".

## Development

`pyproject.toml` uses `uv` (or plain pip) for the dependency lock:

```bash
cd harness/
uv sync                # or: pip install -e .
uv run pytest          # or: pytest
```
