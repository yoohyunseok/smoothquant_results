"""
원본 smoothquant/ppl_eval.py 를 기반으로 한 Colab GPU용 평가 스크립트.
원본 파일은 수정하지 않고 이 복사본만 사용한다.

원본 대비 변경점:
- device: cuda 고정, dtype: --dtype 인자로 fp32/bf16/fp16 선택 (기본 fp16, T4는 bf16 미지원이라 fp16 사용)
- weight_quant / act_quant 를 인자로 받음 (원본은 per_channel/per_token 고정)
- 모델을 바로 GPU(cuda:0)에 올림 (device_map={"": 0}, low_cpu_mem_usage=True) - CPU RAM 경유 최소화
- 결과를 jsonl 한 줄로 append (--results_jsonl), --tag 로 설정 식별
- EVAL_META 한 줄을 stdout에 출력 (run_all.sh 가 파싱해서 평가량 자동 결정에 사용)
- 평가 로직(2048 윈도우, shift CrossEntropy, exp(sum nll/(n*2048)))과
  quantize_bmm_input=True 는 원본과 동일하게 유지
"""
import argparse
import json
import os
import time
import traceback

import torch
import torch.nn as nn
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from smoothquant.fake_quant import quantize_model
from smoothquant.smooth import smooth_lm

DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--act_scales_path", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp32", "bf16", "fp16"]
    )
    parser.add_argument(
        "--act_quant", type=str, default="per_token", choices=["per_token", "per_tensor"]
    )
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="per_channel",
        choices=["per_channel", "per_tensor"],
    )
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--results_jsonl", type=str, required=True)
    return parser.parse_args()


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=None):
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)
        self.n_samples = n_samples

    @property
    def total_windows(self):
        return self.dataset.size(1) // 2048

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.total_windows
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))
        return ppl, n_samples


def append_result(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    record = {
        "tag": args.tag,
        "model_path": args.model_path,
        "dtype": args.dtype,
        "smooth": bool(args.smooth),
        "alpha": args.alpha if args.smooth else None,
        "quantize": bool(args.quantize),
        "act_quant": args.act_quant if args.quantize else None,
        "weight_quant": args.weight_quant if args.quantize else None,
        "n_samples_requested": args.n_samples,
    }
    t_start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=args.n_samples)
        n_windows_total = evaluator.total_windows

        t_load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        load_sec = time.time() - t_load_start

        if args.smooth:
            if not args.act_scales_path:
                raise ValueError("--smooth 사용 시 --act_scales_path 가 필요합니다")
            act_scales = torch.load(args.act_scales_path)
            smooth_lm(model, act_scales, args.alpha)
        if args.quantize:
            model = quantize_model(
                model,
                weight_quant=args.weight_quant,
                act_quant=args.act_quant,
                quantize_bmm_input=True,
            )

        t_eval_start = time.time()
        ppl, n_used = evaluator.evaluate(model)
        eval_sec = time.time() - t_eval_start
        ppl_value = ppl.item()

        status = "ok"
        if ppl_value != ppl_value or ppl_value in (float("inf"), float("-inf")):
            status = "inf_or_nan"

        record.update(
            {
                "status": status,
                "ppl": ppl_value,
                "n_samples_used": n_used,
                "n_windows_total": n_windows_total,
                "load_sec": round(load_sec, 2),
                "eval_sec": round(eval_sec, 2),
                "total_sec": round(time.time() - t_start, 2),
                "error": None,
            }
        )
        print(
            f"EVAL_META tag={args.tag} status={status} ppl={ppl_value} "
            f"n_used={n_used} n_total={n_windows_total} "
            f"load_sec={load_sec:.2f} eval_sec={eval_sec:.2f}"
        )
        append_result(args.results_jsonl, record)
    except torch.cuda.OutOfMemoryError as e:
        record.update(
            {
                "status": "oom",
                "ppl": None,
                "error": str(e),
                "total_sec": round(time.time() - t_start, 2),
            }
        )
        print(f"EVAL_META tag={args.tag} status=oom error={e}")
        append_result(args.results_jsonl, record)
        raise
    except Exception as e:
        record.update(
            {
                "status": "error",
                "ppl": None,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                "total_sec": round(time.time() - t_start, 2),
            }
        )
        print(f"EVAL_META tag={args.tag} status=error error={e}")
        append_result(args.results_jsonl, record)
        raise


if __name__ == "__main__":
    main()
