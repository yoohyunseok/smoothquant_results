"""results.jsonl -> results_table.md 자동 생성. run_all.sh가 설정 하나 끝날 때마다 호출."""
import json
import sys
from collections import OrderedDict, defaultdict

CONFIG_ORDER = ["fp", "A_naive", "A_smooth_a0.5", "B_naive", "B_smooth_a0.5"]
CONFIG_LABEL = {
    "fp": "FP16 기준",
    "A_naive": "설정A naive (per_channel/per_token)",
    "A_smooth_a0.5": "설정A SmoothQuant a=0.5",
    "B_naive": "설정B naive (per_tensor/per_tensor)",
    "B_smooth_a0.5": "설정B SmoothQuant a=0.5",
}
MODEL_ORDER = ["opt-6.7b", "opt-2.7b", "opt-1.3b", "opt-125m"]


def parse_tag(tag):
    # <model_key>_fp16_<config...>  또는 <model_key>_fp16_A_smooth_a<alpha> (alpha sweep)
    parts = tag.split("_fp16_")
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def main():
    jsonl_path = sys.argv[1]
    out_path = sys.argv[2]

    records = OrderedDict()
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 같은 tag가 여러 번 기록되면 마지막 것을 사용(재시도 등)
                records[rec.get("tag")] = rec
    except FileNotFoundError:
        pass

    by_model = defaultdict(dict)
    for tag, rec in records.items():
        model_key, cfg = parse_tag(tag)
        if model_key is None:
            continue
        by_model[model_key][cfg] = rec

    lines = []
    lines.append("# SmoothQuant Colab 결과표 (자동 생성)\n")
    lines.append("각 설정이 끝날 때마다 run_all.sh가 이 파일을 다시 만든다. 직접 수정하지 말 것.\n")

    trend_rows = []

    model_keys_sorted = [m for m in MODEL_ORDER if m in by_model] + [
        m for m in by_model if m not in MODEL_ORDER
    ]

    for model_key in model_keys_sorted:
        cfgs = by_model[model_key]
        lines.append(f"\n## {model_key}\n")
        lines.append("| 설정 | PPL | 상태 | n_samples | FP16 대비 변화(%) | 소요(s) |")
        lines.append("|---|---|---|---|---|---|")
        fp_ppl = None
        fp_rec = cfgs.get("fp")
        if fp_rec and fp_rec.get("status") == "ok":
            fp_ppl = fp_rec["ppl"]
        for cfg in CONFIG_ORDER:
            rec = cfgs.get(cfg)
            label = CONFIG_LABEL.get(cfg, cfg)
            if rec is None:
                lines.append(f"| {label} | - | 미실행 | - | - | - |")
                continue
            ppl = rec.get("ppl")
            status = rec.get("status")
            n_used = rec.get("n_samples_used", "-")
            total_sec = rec.get("total_sec", "-")
            if ppl is not None and fp_ppl:
                delta = f"{(ppl / fp_ppl - 1) * 100:+.2f}%"
            else:
                delta = "-"
            ppl_str = f"{ppl:.4f}" if isinstance(ppl, (int, float)) else "-"
            lines.append(
                f"| {label} | {ppl_str} | {status} | {n_used} | {delta} | {total_sec} |"
            )
            if cfg in ("A_naive", "A_smooth_a0.5") and ppl is not None and fp_ppl:
                trend_rows.append(
                    (model_key, cfg, ppl, fp_ppl, (ppl / fp_ppl - 1) * 100)
                )

        # alpha sweep 결과가 있으면 별도 표시
        alpha_cfgs = sorted(
            [c for c in cfgs if c.startswith("A_smooth_a") or c.startswith("B_smooth_a")]
        )
        if len(alpha_cfgs) > 2:
            lines.append(f"\n### {model_key} alpha sweep\n")
            lines.append("| 설정 | alpha | PPL | 상태 |")
            lines.append("|---|---|---|---|")
            for c in alpha_cfgs:
                rec = cfgs[c]
                alpha = rec.get("alpha")
                ppl = rec.get("ppl")
                ppl_str = f"{ppl:.4f}" if isinstance(ppl, (int, float)) else "-"
                lines.append(f"| {c} | {alpha} | {ppl_str} | {rec.get('status')} |")

    if trend_rows:
        lines.append("\n## 모델 크기별 naive 열화 추세 (설정A 기준)\n")
        lines.append("| model | 설정 | PPL | FP16 PPL | 변화(%) |")
        lines.append("|---|---|---|---|---|")
        for model_key, cfg, ppl, fp_ppl, delta in trend_rows:
            lines.append(f"| {model_key} | {cfg} | {ppl:.4f} | {fp_ppl:.4f} | {delta:+.2f}% |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
