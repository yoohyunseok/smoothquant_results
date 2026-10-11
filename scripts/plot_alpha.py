"""alpha sweep 결과(results.jsonl)를 읽어 alpha-PPL 그래프를 그린다."""
import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    jsonl_path = sys.argv[1]
    model_key = sys.argv[2]
    out_path = sys.argv[3]

    series = defaultdict(list)  # config_label -> list of (alpha, ppl)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tag = rec.get("tag", "")
            if not tag.startswith(f"{model_key}_fp16_"):
                continue
            if rec.get("status") != "ok" or rec.get("alpha") is None:
                continue
            cfg = tag.split("_fp16_")[1]
            if cfg.startswith("A_smooth_a"):
                series["Config A (per_channel/per_token)"].append((rec["alpha"], rec["ppl"]))
            elif cfg.startswith("B_smooth_a"):
                series["Config B (per_tensor/per_tensor)"].append((rec["alpha"], rec["ppl"]))

    if not series:
        print("경고: 그릴 alpha sweep 데이터가 없습니다.")
        return

    plt.figure(figsize=(6, 4))
    for label, points in series.items():
        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("alpha")
    plt.ylabel("PPL (WikiText-2 test)")
    plt.title(f"{model_key}: alpha vs PPL (SmoothQuant W8A8)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"저장됨: {out_path}")


if __name__ == "__main__":
    main()
