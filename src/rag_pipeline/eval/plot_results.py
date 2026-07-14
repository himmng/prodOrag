"""Plot eval results. Reads summary JSON from the type-specific dir, saves plots there."""
import json, sys, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rag_pipeline.config import cfg

# Map eval type → (summary dir, plots dir)
DIRS = {
    "retrieval": (cfg.RETRIEVAL_SUMMARY_DIR, cfg.RETRIEVAL_PLOTS_DIR),
    "ragas":     (cfg.RAGAS_SUMMARY_DIR, cfg.RAGAS_PLOTS_DIR),
}

def latest_summary(summary_dir):
    files = sorted(glob.glob(f"{summary_dir}/*__summary.json"), key=os.path.getmtime)
    if not files:
        sys.exit(f"No summary files in {summary_dir}")
    return files[-1]

def plot_ragas(summary_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np, json, os
    d = json.load(open(summary_path))
    judges = d["judges"]
    metrics = sorted({m for j in judges.values() for m in j["metrics_mean"]})
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics)); w = 0.8 / max(len(judges), 1)
    for i, (jname, jd) in enumerate(judges.items()):
        vals = [jd["metrics_mean"].get(m, 0) for m in metrics]
        bars = ax.bar(x + i*w, vals, w, label=f"{jname} ({jd['deployment']})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x + w*(len(judges)-1)/2)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
    ax.set_ylim(0, 1.1); ax.set_ylabel("score")
    ax.set_title(f"RAGAS · {d['eval_set']} · gen={d['generator']} · n={d['n_questions']}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    stamp = os.path.basename(summary_path).replace("__summary.json", "")
    out = f"{cfg.RAGAS_PLOTS_DIR}/{stamp}__metrics.png"
    plt.savefig(out, dpi=130); plt.close()
    print("plot saved:", out)
    
def main(eval_type="retrieval", path=None):
    summary_dir, plots_dir = DIRS[eval_type]
    path = path or latest_summary(summary_dir)
    d = json.load(open(path))
    stamp = os.path.basename(path).replace("__summary.json", "")
    print("Plotting:", path)

    # ── Plot 1: category × difficulty grid (hit@k) ──
    grid = d["by_category_difficulty"]
    cats = ["ipc_substantive", "bns_substantive", "cross_reference"]
    diffs = ["easy", "medium", "hard"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cats)); w = 0.25
    colors = {"easy": "#12a37f", "medium": "#0f6fc6", "hard": "#c0392b"}
    for i, diff in enumerate(diffs):
        vals = [grid.get(f"{c}|{diff}", {}).get("hit", 0) for c in cats]
        bars = ax.bar(x + (i-1)*w, vals, w, label=diff, color=colors[diff])
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_"," ") for c in cats])
    ax.set_ylim(0, 1.1); ax.set_ylabel("hit@k")
    ax.set_title(f"Retrieval hit@k by category × difficulty · {d['eval_set']}")
    ax.legend(title="difficulty"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p1 = f"{plots_dir}/grid_{stamp}.png"; plt.savefig(p1, dpi=130); plt.close()
    print("saved:", p1)

    # ── Plot 2: metrics by category ──
    bycat = d["by_category"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cats)); w = 0.25
    mcolors = {"hit": "#0f6fc6", "recall": "#12a37f", "mrr": "#d68910"}
    for i, m in enumerate(["hit", "recall", "mrr"]):
        vals = [bycat.get(c, {}).get(m, 0) for c in cats]
        bars = ax.bar(x + (i-1)*w, vals, w, label=m, color=mcolors[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_"," ") for c in cats])
    ax.set_ylim(0, 1.1); ax.set_ylabel("score")
    ax.set_title(f"Retrieval metrics by category · {d['eval_set']}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = f"{plots_dir}/bycat_{stamp}.png"; plt.savefig(p2, dpi=130); plt.close()
    print("saved:", p2)

    # ── Plot 3: negatives / refusal ──
    neg = d.get("negatives", {})
    if neg and neg.get("by_difficulty"):
        nd = neg["by_difficulty"]
        ks = list(nd.keys()); vals = [nd[k]["correct_empty_rate"] for k in ks]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(ks, vals, color="#6c4bb6")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_ylabel("correct-empty rate")
        ax.set_title(f"Out-of-scope handling · floor={neg.get('refusal_floor')}")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p3 = f"{plots_dir}/negatives_{stamp}.png"; plt.savefig(p3, dpi=130); plt.close()
        print("saved:", p3)

if __name__ == "__main__":
    etype = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    main(etype)