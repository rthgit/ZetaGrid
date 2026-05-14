# Soul v2 Runbook

This is the working plan for the next RTH-LM Soul sequence:

1. `text_v2`
2. `code_v2`
3. `math_v1`
4. `70b_fractal_research`

The next consolidation sequence is:

1. `text_align_v1`, `code_align_v1`, `math_align_v1`
2. `instruction_v1`
3. `agentic_v1`
4. `orchestrator_v1`
5. fixed-prompt SwarmLM evaluation

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
python TRAIN_SOUL_V2_FRO_A40.py \
  --mode text_v2 \
  --base_dir /workspace/zetagrid_50b \
  --genome /workspace/zetagrid_50b/zetagrid_25b_production.npy \
  --init_ckpt /workspace/zetagrid_50b/zeta25b_v4_expanded_FINAL.pt \
  --data /workspace/zetagrid_50b/data/text_v2_fineweb_smoke.bin \
  --save_dir /workspace/zetagrid_50b/checkpoints/text_v2 \
  --steps 2000 \
  --save_every 250 \
  --seq_len 256 \
  --batch_size 1 \
  --grad_accum 4 \
  --rank 512 \
  --lr 1e-5
```

Code v2:

```bash
python TRAIN_SOUL_V2_FRO_A40.py \
  --mode code_v2 \
  --base_dir /workspace/zetagrid_50b \
  --genome /workspace/zetagrid_50b/zetagrid_25b_production.npy \
  --init_ckpt /workspace/zetagrid_50b/zeta25b_code_FINAL.pt \
  --data /workspace/zetagrid_50b/data/code_v2/code_v2_5gb.bin \
  --save_dir /workspace/zetagrid_50b/checkpoints/code_v2 \
  --steps 1000 \
  --save_every 250 \
  --seq_len 256 \
  --batch_size 1 \
  --grad_accum 4 \
  --rank 512 \
  --lr 1e-5
```

Math v1:

```bash
python TRAIN_SOUL_V2_FRO_A40.py \
  --mode math_v1 \
  --base_dir /workspace/zetagrid_50b \
  --genome /workspace/zetagrid_50b/zetagrid_25b_production.npy \
  --init_ckpt /workspace/zetagrid_50b/checkpoints/text_v2/TEXT_V2_BEST_0p9111.pt \
  --data /workspace/zetagrid_50b/data/math_v1/math_v1.bin \
  --save_dir /workspace/zetagrid_50b/checkpoints/math_v1 \
  --steps 1000 \
  --save_every 250 \
  --seq_len 256 \
  --batch_size 1 \
  --grad_accum 4 \
  --rank 512 \
  --lr 6e-6 \
  --fro_gamma 0.7
```

## Current Smoke Results

| Soul | Dataset | Checkpoint | Notes |
| --- | --- | --- | --- |
| Text v2 | `/workspace/zetagrid_50b/data/text_v2_fineweb_smoke.bin` (~10GB) | `/workspace/zetagrid_50b/checkpoints/text_v2/TEXT_V2_BEST_0p9111.pt` | Best observed loss 0.9111 around step 1010. |
| Code v2 | `/workspace/zetagrid_50b/data/code_v2/code_v2_5gb.bin` (~5.37GB) | `/workspace/zetagrid_50b/checkpoints/code_v2/latest.pt` | Best tracker was inherited in the first run; observed losses reached about 1.61-1.65. Save as smoke, not release. |

The trainer overwrites `latest.pt` by default and does not write `FINAL.pt` unless `--write_final` is passed. This avoids filling the pod during iterative Soul runs.

## Alignment and SwarmLM Smoke

Build small controlled datasets:

```bash
python BUILD_ALIGN_SWARMLM_DATASETS.py --base_dir /workspace/zetagrid_50b --target_mb 256
```

Align existing Souls with low LR:

```bash
python TRAIN_SOUL_V2_FRO_A40.py --mode text_align_v1 --base_dir /workspace/zetagrid_50b --init_ckpt /workspace/zetagrid_50b/checkpoints/text_v2/TEXT_V2_BEST_0p9111.pt --data /workspace/zetagrid_50b/data/align_v1/text_align_v1.bin --save_dir /workspace/zetagrid_50b/checkpoints/text_align_v1 --steps 500 --save_every 250 --seq_len 384 --batch_size 1 --grad_accum 4 --rank 512 --lr 2e-6 --fro_gamma 0.6
python TRAIN_SOUL_V2_FRO_A40.py --mode code_align_v1 --base_dir /workspace/zetagrid_50b --init_ckpt /workspace/zetagrid_50b/checkpoints/code_v2/CODE_V2_SMOKE.pt --data /workspace/zetagrid_50b/data/align_v1/code_align_v1.bin --save_dir /workspace/zetagrid_50b/checkpoints/code_align_v1 --steps 500 --save_every 250 --seq_len 384 --batch_size 1 --grad_accum 4 --rank 512 --lr 2e-6 --fro_gamma 0.6
python TRAIN_SOUL_V2_FRO_A40.py --mode math_align_v1 --base_dir /workspace/zetagrid_50b --init_ckpt /workspace/zetagrid_50b/checkpoints/math_v1/MATH_V1_SMOKE.pt --data /workspace/zetagrid_50b/data/align_v1/math_align_v1.bin --save_dir /workspace/zetagrid_50b/checkpoints/math_align_v1 --steps 500 --save_every 250 --seq_len 384 --batch_size 1 --grad_accum 4 --rank 512 --lr 2e-6 --fro_gamma 0.7
```

Functional Souls for SwarmLM:

```bash
python TRAIN_SOUL_V2_FRO_A40.py --mode instruction_v1 --base_dir /workspace/zetagrid_50b --init_ckpt /workspace/zetagrid_50b/checkpoints/text_align_v1/TEXT_V2_ALIGN.pt --data /workspace/zetagrid_50b/data/swarmlm_v1/instruction_v1.bin --save_dir /workspace/zetagrid_50b/checkpoints/instruction_v1 --steps 500 --save_every 250 --seq_len 384 --batch_size 1 --grad_accum 4 --rank 512 --lr 2e-6 --fro_gamma 0.6
python TRAIN_SOUL_V2_FRO_A40.py --mode agentic_v1 --base_dir /workspace/zetagrid_50b --init_ckpt /workspace/zetagrid_50b/checkpoints/instruction_v1/INSTRUCTION_V1_SMOKE.pt --data /workspace/zetagrid_50b/data/swarmlm_v1/agentic_v1.bin --save_dir /workspace/zetagrid_50b/checkpoints/agentic_v1 --steps 500 --save_every 250 --seq_len 384 --batch_size 1 --grad_accum 4 --rank 512 --lr 2e-6 --fro_gamma 0.6
python TRAIN_SOUL_V2_FRO_A40.py --mode orchestrator_v1 --base_dir /workspace/zetagrid_50b --init_ckpt /workspace/zetagrid_50b/checkpoints/instruction_v1/INSTRUCTION_V1_SMOKE.pt --data /workspace/zetagrid_50b/data/swarmlm_v1/orchestrator_v1.bin --save_dir /workspace/zetagrid_50b/checkpoints/orchestrator_v1 --steps 500 --save_every 250 --seq_len 384 --batch_size 1 --grad_accum 4 --rank 512 --lr 2e-6 --fro_gamma 0.6
```

Run scientific smoke evaluation:

```bash
python EVAL_SWARMLM_SUITE.py \
  --base_dir /workspace/zetagrid_50b \
  --out_dir /workspace/zetagrid_50b/reports/swarmlm_v1_suite \
  --max_new 120 \
  --temperature 0.25 \
  --top_k 10
```

For full checkpoint provenance, add `--hash_files`. Hashing the Genome and six 3.6GB Soul checkpoints is slower but useful before publishing a report.

## Logging Requirements

Every run should preserve:

- `fro_metrics.jsonl`
- checkpoint path and hash
- dataset manifest and byte size
- fixed prompt outputs at checkpoints
- VRAM peak and effective tokens/sec

Do not publish a new Soul only from loss. Require fixed prompt output and a small held-out eval before uploading.
