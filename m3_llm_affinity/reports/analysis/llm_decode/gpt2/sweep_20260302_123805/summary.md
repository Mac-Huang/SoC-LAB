# Suite Summary

## What We Ran

- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260228_234558_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260228_234623_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260228_234817_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260228_234902_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260228_234916_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260301_002127_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260301_002210_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260301_002223_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260301_002348_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260301_002431_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260301_002445_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260302_123555_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260302_123709_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260302_123731_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260302_123805_gpt2_ctx64_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260302_123805_gpt2_ctx128_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260302_123805_gpt2_ctx256_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260302_123805_gpt2_ctx512_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260302_123805_gpt2_ctx1024_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260302_123805_gpt2_ctx2048_bench.jsonl`
- `/Users/Patron/Desktop/Develop/SoC-Lab/m3_llm_affinity/results/20260302_123805_gpt2_ctx4096_bench.jsonl`

Detected models: `gpt2`

Detected context lengths: `64, 128, 256, 512, 1024, 2048, 4096`

![All models TTFT vs Throughput](fig_all_models_ttft_vs_throughput.png)

## Top-3 Fastest TTFT Per Model

### gpt2
| context_len | scenario_label | ttft_ms_mean |
| --- | --- | --- |
| 64 | NE | 12.041 |
| 64 | NE→GPU | 14.018 |
| 64 | ALL | 14.069 |

## Top-3 Throughput Per Model

### gpt2
| context_len | scenario_label | tokens_per_sec_mean |
| --- | --- | --- |
| 64 | GPU→NE | 149.017 |
| 64 | CPU | 146.713 |
| 64 | NE | 146.340 |

## Tradeoff Note

- gpt2: ctx 512: TTFT winner=NE→GPU, throughput winner=CPU

## Error Summary

| model_alias | context_len | scenario_label | error_count |
| --- | --- | --- | --- |
| gpt2 | 2048 | unknown | 1 |
| gpt2 | 4096 | unknown | 1 |

## Figures

### gpt2
![fig_gpt2_ttft_ms](fig_gpt2_ttft_ms.png)
![fig_gpt2_tokens_per_sec](fig_gpt2_tokens_per_sec.png)
![fig_gpt2_tflops_prefill](fig_gpt2_tflops_prefill.png)
![fig_gpt2_tflops_decode](fig_gpt2_tflops_decode.png)
![fig_gpt2_peak_rss_mb](fig_gpt2_peak_rss_mb.png)
