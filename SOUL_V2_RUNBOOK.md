# Soul v2 Runbook

This is the working plan for the next RTH-LM Soul sequence:

1. `text_v2`
2. `code_v2`
3. `math_v1`
4. `70b_fractal_research`

The first three runs reuse the existing 25B Genome/Soul assets:

| Role | Local artifact |
| --- | --- |
| Genome | `zetagrid_25b_production.npy` |
| Text init Soul | `zeta25b_v4_expanded_FINAL.pt` |
| Code init Soul | `zeta25b_code_FINAL.pt` |
| Text GGUF | `rth_lm_25b_v4.gguf` |
| Code GGUF | `rth_lm_25b_code.gguf` |

## Dataset Targets

| Soul | Smoke | First serious run | Full run |
| --- | ---: | ---: | ---: |
| Text v2 | 5-10GB | 50-100GB | 150-300GB |
| Code v2 | 5-10GB | 100GB | 200-300GB |
| Math v1 | 1-5GB | 20-50GB | 80GB |

Text v2 should start with FineWeb/FineWeb-Edu. Use FineWeb-Edu for quality and FineWeb for volume. Keep Italian/EU/technical text as a smaller controlled mix after the first smoke run.

## A40 Smoke Commands

Text v2:

```bash
python TRAIN_SOUL_V2_FRO_A40.py --mode text_v2 --data /workspace/data/text_v2/fineweb_text_v2.bin --steps 500 --seq_len 512 --batch_size 1 --grad_accum 16 --rank 512
```

Code v2:

```bash
python TRAIN_SOUL_V2_FRO_A40.py --mode code_v2 --data /workspace/data/code_v2/code_v2.bin --steps 500 --seq_len 512 --batch_size 1 --grad_accum 16 --rank 512
```

Math v1:

```bash
python TRAIN_SOUL_V2_FRO_A40.py --mode math_v1 --data /workspace/data/math_v1/math_v1.bin --steps 500 --seq_len 512 --batch_size 1 --grad_accum 16 --rank 256 --lr 5e-5
```

## Logging Requirements

Every run should preserve:

- `fro_metrics.jsonl`
- checkpoint path and hash
- dataset manifest and byte size
- fixed prompt outputs at checkpoints
- VRAM peak and effective tokens/sec

Do not publish a new Soul only from loss. Require fixed prompt output and a small held-out eval before uploading.
