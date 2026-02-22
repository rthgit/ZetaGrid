#!/usr/bin/env python3
"""
MOBILE_ONNX_PRE_RELEASE_TEST.py
===============================
Pre-release validation for mobile ONNX exports.

Checks performed:
1. Session load and inference smoke test.
2. First-token latency and decode throughput.
3. Numerical stability (finite logits).
4. Simple generation health metrics.
5. JSON report with release gate verdict.

Example:
    python MOBILE_ONNX_PRE_RELEASE_TEST.py \
      --model E:/ZETAGRID/rth_lm_25b_v4.mid3b.onnx \
      --model E:/ZETAGRID/rth_lm_25b_code.mid3b.onnx \
      --max-new-tokens 48 \
      --runs-per-prompt 2 \
      --max-first-token-ms 1200 \
      --min-decode-tps 4.0 \
      --require-pass
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None


SCRIPT_VERSION = "1.1.0"

ASCII_ALLOWED_TOKENS = np.asarray([9, 10, 13] + list(range(32, 127)), dtype=np.int64)

DEFAULT_PROMPTS = [
    "Ciao, spiegami in due frasi cos'e un modello linguistico.",
    "Write a short Python function that checks if a string is a palindrome.",
    "The future of efficient AI on mobile devices is",
    "Dammi 3 idee startup AI low-cost in Italia.",
    "def quicksort(arr):",
]


def utc_now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * p))
    return float(arr[idx])


def longest_run(tokens: List[int]) -> int:
    if not tokens:
        return 0
    best = 1
    cur = 1
    last = tokens[0]
    for tok in tokens[1:]:
        if tok == last:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
            last = tok
    return best


def entropy_from_logits(logits: np.ndarray) -> float:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)
    denom = float(np.sum(exp_x))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    p = exp_x / denom
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    y = y - np.max(y)
    exp_y = np.exp(y)
    denom = float(np.sum(exp_y))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros_like(y, dtype=np.float64)
    return exp_y / denom


def pick_next_token(
    logits: np.ndarray,
    generated: List[int],
    rng: np.random.Generator,
    sampling_mode: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    recent_window: int,
    constrain_ascii: bool,
    extra_bias: Optional[np.ndarray],
) -> int:
    scores = np.asarray(logits, dtype=np.float64).copy()
    if scores.ndim != 1:
        scores = scores.reshape(-1)

    if extra_bias is not None:
        eb = np.asarray(extra_bias, dtype=np.float64).reshape(-1)
        if eb.shape[0] == scores.shape[0]:
            scores += eb

    if repetition_penalty > 1.0 and generated:
        recent = generated[-max(1, int(recent_window)) :]
        for tok in set(recent):
            if 0 <= tok < scores.shape[0]:
                if scores[tok] >= 0:
                    scores[tok] /= repetition_penalty
                else:
                    scores[tok] *= repetition_penalty

    if constrain_ascii:
        masked = np.full_like(scores, -np.inf)
        allowed = ASCII_ALLOWED_TOKENS[ASCII_ALLOWED_TOKENS < scores.shape[0]]
        masked[allowed] = scores[allowed]
        scores = masked

    if sampling_mode == "greedy" or temperature <= 0:
        return int(np.argmax(scores))

    scores = scores / max(1e-6, float(temperature))

    if top_k > 0 and top_k < scores.shape[0]:
        idx = np.argpartition(scores, -top_k)[-top_k:]
        k_mask = np.full_like(scores, -np.inf)
        k_mask[idx] = scores[idx]
        scores = k_mask

    probs = _softmax_1d(scores)
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        return int(np.argmax(scores))

    if 0.0 < top_p < 1.0:
        sort_idx = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[sort_idx])
        keep_n = int(np.searchsorted(cum, top_p, side="right")) + 1
        keep_n = max(1, min(keep_n, probs.shape[0]))
        keep = sort_idx[:keep_n]
        p_mask = np.zeros_like(probs)
        p_mask[keep] = probs[keep]
        norm = float(np.sum(p_mask))
        if norm > 0:
            probs = p_mask / norm

    return int(rng.choice(probs.shape[0], p=probs))


def load_prompts(prompts_file: Optional[str]) -> List[str]:
    if not prompts_file:
        return DEFAULT_PROMPTS

    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    prompts: List[str] = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)

    if not prompts:
        raise ValueError("Prompts file is empty.")
    return prompts


class MobileOnnxRunner:
    def __init__(
        self,
        model_path: str,
        providers: List[str],
        threads: int,
        max_seq_len: int,
        pad_byte: int,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            ) from exc

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = model_path
        self.max_seq_len = max(8, int(max_seq_len))
        self.pad_byte = int(max(0, min(255, pad_byte)))

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = max(1, int(threads) // 2)

        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.active_providers = self.session.get_providers()

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError("Invalid ONNX graph: missing inputs/outputs.")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_shape = inputs[0].shape

        self.fixed_batch: Optional[int] = None
        self.fixed_seq: Optional[int] = None

        if len(self.input_shape) >= 1 and isinstance(self.input_shape[0], int):
            self.fixed_batch = int(self.input_shape[0])
        if len(self.input_shape) >= 2 and isinstance(self.input_shape[1], int):
            self.fixed_seq = int(self.input_shape[1])

        self._proc = psutil.Process(os.getpid()) if psutil is not None else None

    def memory_mb(self) -> float:
        if self._proc is None:
            return 0.0
        return float(self._proc.memory_info().rss / (1024 * 1024))

    def _prepare_input(self, token_ids: List[int]) -> np.ndarray:
        if self.fixed_seq is not None:
            seq_len = self.fixed_seq
        else:
            seq_len = min(self.max_seq_len, len(token_ids))
            seq_len = max(1, seq_len)

        data = token_ids[-seq_len:]
        if len(data) < seq_len:
            data = [self.pad_byte] * (seq_len - len(data)) + data

        arr = np.asarray([data], dtype=np.int64)

        if self.fixed_batch is not None and self.fixed_batch > 1:
            arr = np.repeat(arr, self.fixed_batch, axis=0)

        return arr

    def infer(self, token_ids: List[int]) -> Dict[str, Any]:
        inp = self._prepare_input(token_ids)
        t0 = time.perf_counter()
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        ms = (time.perf_counter() - t0) * 1000.0

        logits = np.asarray(out)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        elif logits.ndim == 2:
            pass
        elif logits.ndim == 3:
            logits = logits[:, -1, :]
        else:
            logits = logits.reshape(logits.shape[0], -1)

        return {"logits": logits, "latency_ms": float(ms)}


def run_single_case(
    runner: MobileOnnxRunner,
    prompt: str,
    run_idx: int,
    max_new_tokens: int,
    warmup_calls: int,
    sampling_mode: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    recent_window: int,
    constrain_ascii: bool,
    seed: int,
    bigram_logprobs: Optional[np.ndarray],
    bigram_lambda: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "prompt": prompt,
        "run_index": run_idx,
        "status": "ok",
        "error": None,
    }

    token_ids = list(prompt.encode("utf-8", errors="ignore"))
    if not token_ids:
        token_ids = [32]

    mem_before = runner.memory_mb()
    mem_peak = mem_before

    try:
        rng = np.random.default_rng(seed + (run_idx * 104729))

        for _ in range(max(0, warmup_calls)):
            _ = runner.infer(token_ids)
            mem_peak = max(mem_peak, runner.memory_mb())

        # First token latency (prefill)
        first = runner.infer(token_ids)
        first_logits = first["logits"]
        first_ms = float(first["latency_ms"])
        mem_peak = max(mem_peak, runner.memory_mb())

        generated: List[int] = []
        decode_latencies: List[float] = []
        entropy_values: List[float] = []
        invalid_steps = 0

        if not np.isfinite(first_logits).all():
            invalid_steps += 1
            raise RuntimeError("Non-finite logits at first token.")

        prev_tok0 = token_ids[-1] if token_ids else None
        extra_bias0 = None
        if (
            bigram_logprobs is not None
            and prev_tok0 is not None
            and 0 <= prev_tok0 < bigram_logprobs.shape[0]
            and bigram_lambda != 0.0
        ):
            extra_bias0 = float(bigram_lambda) * bigram_logprobs[prev_tok0]

        next_tok = pick_next_token(
            logits=first_logits[0],
            generated=generated,
            rng=rng,
            sampling_mode=sampling_mode,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            recent_window=recent_window,
            constrain_ascii=constrain_ascii,
            extra_bias=extra_bias0,
        )
        generated.append(next_tok)
        token_ids.append(next_tok)
        entropy_values.append(entropy_from_logits(first_logits[0]))

        for _ in range(max(0, max_new_tokens - 1)):
            out = runner.infer(token_ids)
            logits = out["logits"]
            dt = float(out["latency_ms"])
            decode_latencies.append(dt)
            mem_peak = max(mem_peak, runner.memory_mb())

            if not np.isfinite(logits).all():
                invalid_steps += 1
                raise RuntimeError("Non-finite logits during decode.")

            prev_tok = token_ids[-1] if token_ids else None
            extra_bias = None
            if (
                bigram_logprobs is not None
                and prev_tok is not None
                and 0 <= prev_tok < bigram_logprobs.shape[0]
                and bigram_lambda != 0.0
            ):
                extra_bias = float(bigram_lambda) * bigram_logprobs[prev_tok]

            tok = pick_next_token(
                logits=logits[0],
                generated=generated,
                rng=rng,
                sampling_mode=sampling_mode,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                recent_window=recent_window,
                constrain_ascii=constrain_ascii,
                extra_bias=extra_bias,
            )
            generated.append(tok)
            token_ids.append(tok)
            entropy_values.append(entropy_from_logits(logits[0]))

        generated_text = bytes(generated).decode("utf-8", errors="replace")
        replacement_count = generated_text.count("\ufffd")
        replacement_ratio = float(replacement_count / max(1, len(generated_text)))

        unique_ratio = float(len(set(generated)) / max(1, len(generated)))
        repetition_ratio = float(1.0 - unique_ratio)
        longest_repeat = int(longest_run(generated))

        decode_seconds = float(sum(decode_latencies) / 1000.0)
        decode_tokens = max(0, len(generated) - 1)
        decode_tps = float(decode_tokens / decode_seconds) if decode_seconds > 0 and decode_tokens > 0 else 0.0

        overall_seconds = float((first_ms / 1000.0) + decode_seconds)
        overall_tps = float(len(generated) / overall_seconds) if overall_seconds > 0 else 0.0

        result.update(
            {
                "first_token_ms": first_ms,
                "decode_avg_ms": safe_mean(decode_latencies),
                "decode_p95_ms": percentile(decode_latencies, 0.95),
                "decode_tps": decode_tps,
                "overall_tps": overall_tps,
                "generated_tokens": len(generated),
                "invalid_steps": invalid_steps,
                "entropy_avg": safe_mean(entropy_values),
                "entropy_min": float(min(entropy_values) if entropy_values else 0.0),
                "replacement_ratio": replacement_ratio,
                "repetition_ratio": repetition_ratio,
                "longest_repeat_run": longest_repeat,
                "sample_output": generated_text[:240],
                "memory_mb_before": float(mem_before),
                "memory_mb_peak": float(mem_peak),
                "sampling_mode": sampling_mode,
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "recent_window": int(recent_window),
                "constrain_ascii": bool(constrain_ascii),
                "bigram_lambda": float(bigram_lambda),
            }
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result.update(
            {
                "first_token_ms": 0.0,
                "decode_avg_ms": 0.0,
                "decode_p95_ms": 0.0,
                "decode_tps": 0.0,
                "overall_tps": 0.0,
                "generated_tokens": 0,
                "invalid_steps": 1,
                "entropy_avg": 0.0,
                "entropy_min": 0.0,
                "replacement_ratio": 1.0,
                "repetition_ratio": 1.0,
                "longest_repeat_run": 0,
                "sample_output": "",
                "memory_mb_before": float(mem_before),
                "memory_mb_peak": float(mem_peak),
                "sampling_mode": sampling_mode,
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "recent_window": int(recent_window),
                "constrain_ascii": bool(constrain_ascii),
                "bigram_lambda": float(bigram_lambda),
            }
        )

    return result


def summarize_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cases)
    successes = [c for c in cases if c.get("status") == "ok"]
    failures = [c for c in cases if c.get("status") != "ok"]

    return {
        "total_runs": total,
        "successful_runs": len(successes),
        "failed_runs": len(failures),
        "fail_rate": float(len(failures) / total) if total > 0 else 1.0,
        "avg_first_token_ms": safe_mean([float(c["first_token_ms"]) for c in successes]),
        "p95_first_token_ms": percentile([float(c["first_token_ms"]) for c in successes], 0.95),
        "avg_decode_tps": safe_mean([float(c["decode_tps"]) for c in successes]),
        "avg_overall_tps": safe_mean([float(c["overall_tps"]) for c in successes]),
        "avg_entropy": safe_mean([float(c["entropy_avg"]) for c in successes]),
        "avg_repetition_ratio": safe_mean([float(c["repetition_ratio"]) for c in successes]),
        "avg_replacement_ratio": safe_mean([float(c["replacement_ratio"]) for c in successes]),
        "max_memory_mb_peak": float(max([float(c["memory_mb_peak"]) for c in successes], default=0.0)),
        "total_invalid_steps": int(sum(int(c["invalid_steps"]) for c in cases)),
    }


def gate_release(
    summary: Dict[str, Any],
    max_fail_rate: float,
    max_first_token_ms: Optional[float],
    min_decode_tps: Optional[float],
    max_invalid_steps: int,
    max_replacement_ratio: Optional[float],
    max_repetition_ratio: Optional[float],
) -> Dict[str, Any]:
    reasons: List[str] = []

    if summary["fail_rate"] > max_fail_rate:
        reasons.append(
            f"fail_rate {summary['fail_rate']:.3f} > allowed {max_fail_rate:.3f}"
        )

    if max_first_token_ms is not None and summary["avg_first_token_ms"] > max_first_token_ms:
        reasons.append(
            f"avg_first_token_ms {summary['avg_first_token_ms']:.2f} > allowed {max_first_token_ms:.2f}"
        )

    if min_decode_tps is not None and summary["avg_decode_tps"] < min_decode_tps:
        reasons.append(
            f"avg_decode_tps {summary['avg_decode_tps']:.3f} < required {min_decode_tps:.3f}"
        )

    if summary["total_invalid_steps"] > max_invalid_steps:
        reasons.append(
            f"total_invalid_steps {summary['total_invalid_steps']} > allowed {max_invalid_steps}"
        )

    if (
        max_replacement_ratio is not None
        and summary["avg_replacement_ratio"] > max_replacement_ratio
    ):
        reasons.append(
            f"avg_replacement_ratio {summary['avg_replacement_ratio']:.3f} > allowed {max_replacement_ratio:.3f}"
        )

    if (
        max_repetition_ratio is not None
        and summary["avg_repetition_ratio"] > max_repetition_ratio
    ):
        reasons.append(
            f"avg_repetition_ratio {summary['avg_repetition_ratio']:.3f} > allowed {max_repetition_ratio:.3f}"
        )

    return {"pass": len(reasons) == 0, "reasons": reasons}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-release test for mobile ONNX models.")
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        help="Path to ONNX model. Repeat --model for multiple files.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Optional text file with one prompt per line.",
    )
    parser.add_argument("--runs-per-prompt", type=int, default=1, help="How many runs for each prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Generated tokens per run.")
    parser.add_argument("--warmup-calls", type=int, default=1, help="Warmup inferences per run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--threads", type=int, default=0, help="ONNX Runtime threads (0 = runtime default).")
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ORT providers (default CPUExecutionProvider).",
    )
    parser.add_argument("--max-seq-len", type=int, default=256, help="Context cap for dynamic-shape models.")
    parser.add_argument("--pad-byte", type=int, default=32, help="Left-pad byte value for short fixed-shape inputs.")
    parser.add_argument(
        "--sampling-mode",
        choices=["greedy", "sample"],
        default="sample",
        help="Token selection strategy.",
    )
    parser.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k cutoff for sampling (0 disables).")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus cutoff for sampling.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Penalty (>1) applied to recently generated tokens.",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=64,
        help="How many recent tokens are penalized for repetition.",
    )
    parser.add_argument(
        "--constrain-ascii",
        action="store_true",
        help="Restrict generated tokens to ASCII printable bytes plus whitespace controls.",
    )
    parser.add_argument(
        "--bigram-adapter",
        default=None,
        help="Optional .npz adapter containing key 'log_bigram' (256x256).",
    )
    parser.add_argument(
        "--bigram-lambda",
        type=float,
        default=0.0,
        help="Weight for bigram adapter bias (0 disables).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output JSON report path. Default: mobile_pre_release_report_<timestamp>.json",
    )

    # Release gates (optional except fail rate)
    parser.add_argument("--max-fail-rate", type=float, default=0.0, help="Maximum allowed run failure ratio.")
    parser.add_argument("--max-first-token-ms", type=float, default=None, help="Maximum allowed average first-token latency.")
    parser.add_argument("--min-decode-tps", type=float, default=None, help="Minimum allowed average decode tokens/sec.")
    parser.add_argument("--max-invalid-steps", type=int, default=0, help="Maximum allowed non-finite logits steps.")
    parser.add_argument(
        "--max-replacement-ratio",
        type=float,
        default=None,
        help="Maximum allowed average replacement-char ratio (U+FFFD).",
    )
    parser.add_argument(
        "--max-repetition-ratio",
        type=float,
        default=None,
        help="Maximum allowed average repetition ratio (1 - unique_token_ratio).",
    )
    parser.add_argument("--require-pass", action="store_true", help="Exit with code 2 if release gate fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        prompts = load_prompts(args.prompts_file)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    if not providers:
        providers = ["CPUExecutionProvider"]

    bigram_logprobs: Optional[np.ndarray] = None
    if args.bigram_adapter:
        if not os.path.exists(args.bigram_adapter):
            print(f"[ERROR] Bigram adapter not found: {args.bigram_adapter}")
            return 1
        try:
            pack = np.load(args.bigram_adapter)
            bigram_logprobs = np.asarray(pack["log_bigram"], dtype=np.float32)
            if bigram_logprobs.shape != (256, 256):
                raise ValueError(f"Expected log_bigram shape (256,256), got {bigram_logprobs.shape}")
        except Exception as exc:
            print(f"[ERROR] Failed to load bigram adapter: {exc}")
            return 1

    report: Dict[str, Any] = {
        "script": "MOBILE_ONNX_PRE_RELEASE_TEST.py",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now_iso(),
        "args": vars(args),
        "prompts_count": len(prompts),
        "models": [],
    }

    all_pass = True

    for model_path in args.models:
        print(f"\n[MODEL] {model_path}")
        model_entry: Dict[str, Any] = {
            "model_path": model_path,
            "session": {},
            "cases": [],
            "summary": {},
            "gate": {},
        }

        try:
            runner = MobileOnnxRunner(
                model_path=model_path,
                providers=providers,
                threads=int(args.threads),
                max_seq_len=int(args.max_seq_len),
                pad_byte=int(args.pad_byte),
            )

            model_entry["session"] = {
                "providers_requested": providers,
                "providers_active": runner.active_providers,
                "input_name": runner.input_name,
                "input_shape": runner.input_shape,
                "output_name": runner.output_name,
            }

            run_id = 0
            for prompt in prompts:
                for _ in range(max(1, int(args.runs_per_prompt))):
                    run_id += 1
                    case = run_single_case(
                        runner=runner,
                        prompt=prompt,
                        run_idx=run_id,
                        max_new_tokens=max(1, int(args.max_new_tokens)),
                        warmup_calls=max(0, int(args.warmup_calls)),
                        sampling_mode=str(args.sampling_mode),
                        temperature=float(args.temperature),
                        top_k=max(0, int(args.top_k)),
                        top_p=float(args.top_p),
                        repetition_penalty=float(args.repetition_penalty),
                        recent_window=max(1, int(args.recent_window)),
                        constrain_ascii=bool(args.constrain_ascii),
                        seed=int(args.seed),
                        bigram_logprobs=bigram_logprobs,
                        bigram_lambda=float(args.bigram_lambda),
                    )
                    model_entry["cases"].append(case)
                    status = case["status"]
                    ft = case["first_token_ms"]
                    dtps = case["decode_tps"]
                    print(
                        f"  run={run_id:03d} status={status} first_ms={ft:.2f} decode_tps={dtps:.3f}"
                    )

            summary = summarize_cases(model_entry["cases"])
            gate = gate_release(
                summary=summary,
                max_fail_rate=float(args.max_fail_rate),
                max_first_token_ms=args.max_first_token_ms,
                min_decode_tps=args.min_decode_tps,
                max_invalid_steps=int(args.max_invalid_steps),
                max_replacement_ratio=args.max_replacement_ratio,
                max_repetition_ratio=args.max_repetition_ratio,
            )

            model_entry["summary"] = summary
            model_entry["gate"] = gate

            print(
                "  summary: "
                f"fail_rate={summary['fail_rate']:.3f}, "
                f"avg_first_ms={summary['avg_first_token_ms']:.2f}, "
                f"avg_decode_tps={summary['avg_decode_tps']:.3f}, "
                f"avg_repl={summary['avg_replacement_ratio']:.3f}, "
                f"avg_rep={summary['avg_repetition_ratio']:.3f}"
            )
            print(f"  gate: {'PASS' if gate['pass'] else 'FAIL'}")
            if not gate["pass"]:
                for reason in gate["reasons"]:
                    print(f"    reason: {reason}")

            all_pass = all_pass and bool(gate["pass"])
        except Exception as exc:
            model_entry["summary"] = {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "fail_rate": 1.0,
                "avg_first_token_ms": 0.0,
                "p95_first_token_ms": 0.0,
                "avg_decode_tps": 0.0,
                "avg_overall_tps": 0.0,
                "avg_entropy": 0.0,
                "avg_repetition_ratio": 0.0,
                "avg_replacement_ratio": 0.0,
                "max_memory_mb_peak": 0.0,
                "total_invalid_steps": 1,
            }
            model_entry["gate"] = {"pass": False, "reasons": [str(exc)]}
            all_pass = False
            print(f"  gate: FAIL")
            print(f"    reason: {exc}")

        report["models"].append(model_entry)

    report["overall"] = {
        "all_passed": all_pass,
        "tested_models": len(report["models"]),
    }

    if args.report:
        report_path = args.report
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = f"mobile_pre_release_report_{ts}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[REPORT] {report_path}")
    print(f"[RELEASE] {'PASS' if all_pass else 'FAIL'}")

    if args.require_pass and not all_pass:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
