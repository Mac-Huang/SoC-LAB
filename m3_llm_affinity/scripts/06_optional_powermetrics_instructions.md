# Optional Power Monitoring (Manual)

These commands are optional and manual. The benchmark scripts do not require sudo and do not automate power collection.

## 1) Create a log folder

```bash
mkdir -p reports/power
```

## 2) Run benchmark in one terminal

```bash
make bench
```

## 3) Collect powermetrics in another terminal (requires sudo)

```bash
sudo powermetrics \
  --samplers cpu_power,gpu_power,ane_power \
  --show-process-energy \
  --sample-rate 1000 \
  -i 1 \
  > reports/power/powermetrics_$(date +%Y%m%d_%H%M%S).log
```

Stop with `Ctrl+C` after benchmark completes.

## 4) Optional: collect asitop snapshot

If `asitop` is installed:

```bash
asitop --interval 1 --format csv > reports/power/asitop_$(date +%Y%m%d_%H%M%S).csv
```

Stop with `Ctrl+C` after benchmark completes.

## Notes

- Keep benchmark settings fixed when comparing power traces.
- Record the matching benchmark JSONL filename alongside power logs for traceability.
