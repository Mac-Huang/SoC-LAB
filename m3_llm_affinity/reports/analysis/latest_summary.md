# Benchmark Analysis

## Inputs
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260301_002348_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260301_002431_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260301_002445_bench.jsonl`

## Key Findings
- Best decode throughput: `split:CPU_AND_GPU->CPU_AND_NE` at 148.25 tok/s (95% CI +/- 0.47).
- Slowest decode throughput: `split:CPU_AND_NE->CPU_AND_GPU` at 99.88 tok/s.
- Prefill is fastest on configurations using NE, but end-to-end decode throughput depends more on decode-stage placement.
- Compute plan preference split: prefill has 346 NE-preferred ops; decode has 423 GPU-preferred ops.

## Aggregated Table

| scenario | n_runs | prefill_latency_ms_mean | total_decode_latency_ms_mean | tokens_per_sec_mean | effective_TFLOPS_prefill_mean | effective_TFLOPS_decode_mean | peak_rss_mb_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| split:CPU_AND_GPU->CPU_AND_NE | 20 | 20.08 | 215.856 | 148.254 | 0.549 | 0.026 | 749.484 |
| whole:CPU_ONLY | 20 | 10.397 | 217.437 | 147.173 | 1.044 | 0.025 | 828.506 |
| whole:CPU_AND_NE | 20 | 5.418 | 218.368 | 146.564 | 2.004 | 0.025 | 965.703 |
| whole:CPU_AND_GPU | 20 | 28.289 | 233.475 | 137.101 | 0.388 | 0.024 | 958.292 |
| whole:ALL | 20 | 5.235 | 283.212 | 113.014 | 2.079 | 0.019 | 1048.634 |
| split:CPU_AND_NE->CPU_AND_GPU | 20 | 5.225 | 323.491 | 99.876 | 2.078 | 0.017 | 744.564 |
