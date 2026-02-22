# LIFE COMMON PIPELINE (IT)

## 1) Build datasets
```powershell
python PREPARE_LIFE_COMMON_DATASET.py --train-size 3200 --eval-size 320
```

Outputs:
- `datasets/life_common_it.jsonl`
- `datasets/life_common_eval_it.jsonl`

## 2) Build RAG index (BM25)
```powershell
python BUILD_LIFE_COMMON_RAG.py --input datasets/life_common_it.jsonl --index-out rag/life_common_it_bm25.json
```

Quick test:
```powershell
python BUILD_LIFE_COMMON_RAG.py --skip-build --index-out rag/life_common_it_bm25.json --query "come organizzo bollette e scadenze" --top-k 3
```

## 3) Evaluate
RAG baseline:
```powershell
python EVAL_LIFE_COMMON.py --mode rag --eval-set datasets/life_common_eval_it.jsonl --index rag/life_common_it_bm25.json --limit 200 --report reports/life_common_eval_report.json
```

External model command:
```powershell
python EVAL_LIFE_COMMON.py --mode cmd --eval-set datasets/life_common_eval_it.jsonl --answer-cmd "python my_infer.py --prompt \"{prompt}\""
```

## 4) Convert JSONL to byte-level BIN
Useful for scripts that train on `uint8` streams.
```powershell
python BUILD_LIFE_COMMON_BIN.py --input datasets/life_common_it.jsonl --output datasets/life_common_it.bin --repeat 3
```

Output:
- `datasets/life_common_it.bin`
