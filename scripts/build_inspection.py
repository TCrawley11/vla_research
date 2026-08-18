"""Build <out_dir>/inspection.html: key frames next to each model's captions
and QA so annotations can be eyeballed against the images.

Reads the team-schema result files written by annotate_benchmark.py
(<sample_id>__<model>.json) plus the frames/ directory of key-frame JPEGs.
When every model on a sample answered the same question set (the benchmark's
normal shape) the answers are laid out one question per row, one column per
model; otherwise (legacy per-model question sets) each model gets its own
table.

Usage: uv run python scripts/build_inspection.py [--dir data/annotation_test]
"""

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

CAM_ORDER = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "BACK"]
QA_ORDER = ["perception", "prediction", "planning", "behaviour"]

CSS = """
body { font-family: sans-serif; margin: 1.5rem; background: #16181d; color: #dfe3ea; }
h1 { font-size: 1.3rem; } h2 { font-size: 1.1rem; margin: 0 0 .3rem; }
.sample { border: 1px solid #333a46; border-radius: 8px; padding: 1rem; margin-bottom: 2rem; }
.frames { display: flex; gap: .5rem; margin-bottom: .6rem; }
.frames figure { margin: 0; flex: 1; min-width: 0; }
.frames img { width: 100%; border-radius: 4px; }
.frames figcaption { font-size: .75rem; text-align: center; color: #9aa3b2; }
.gt { font-size: .85rem; color: #9aa3b2; margin-bottom: .8rem; }
.models { display: flex; gap: 1rem; align-items: flex-start; flex-wrap: wrap; }
.model { flex: 1; min-width: 24rem; background: #1d2027; border-radius: 6px; padding: .8rem; }
.model h3 { margin: 0 0 .5rem; font-size: .95rem; color: #8ab4f8; }
.cap { margin: .3rem 0; } .cap b { color: #c6cdd8; }
table { border-collapse: collapse; width: 100%; font-size: .82rem; margin-top: .5rem; }
td, th { border: 1px solid #333a46; padding: .3rem .45rem; vertical-align: top; text-align: left; }
th { color: #8ab4f8; font-weight: 600; }
.qtype { white-space: nowrap; color: #9aa3b2; }
.q { color: #c6cdd8; min-width: 14rem; }
.action { font-size: .82rem; margin-top: .5rem; color: #b7e0a8; }
.meta { font-size: .72rem; color: #6f7885; margin-top: .4rem; }
.summary { margin-bottom: 1.5rem; } .summary table { width: auto; }
"""


def load_results(out_dir: Path):
    by_sample = defaultdict(dict)
    for p in sorted(out_dir.glob("*__*.json")):
        d = json.loads(p.read_text())
        by_sample[d["ground_truth"]["sample_id"]][d["model"]] = d
    return by_sample


def meta_line(d: dict) -> str:
    meta = d.get("meta", {})
    usage = meta.get("usage", {})
    bits = [f"attempts {meta.get('attempts', '?')}",
            f"tokens {usage.get('prompt_tokens', '?')} in / "
            f"{usage.get('completion_tokens', '?')} out"]
    if usage.get("reasoning_tokens"):
        bits.append(f"{usage['reasoning_tokens']} reasoning")
    if usage.get("cost_usd") is not None:
        bits.append(f"${usage['cost_usd']:.4f}")
    if meta.get("provider"):
        bits.append(f"via {meta['provider']}")
    bits.append(f"response_format {meta.get('response_format', '?')}")
    bits.append(f"prompt {d.get('prompt_id', '?')}")
    return html.escape(", ".join(bits))


def question_key(d: dict) -> tuple:
    return tuple((p["type"], p["question"]) for p in d["annotation"]["qa_pairs"])


def render_shared(models: dict) -> str:
    """One table: rows = shared questions, columns = models."""
    names = sorted(models)
    first = models[names[0]]
    head = "".join(f"<th>{html.escape(m)}</th>" for m in names)
    cap_rows = []
    for key in ("caption_short", "caption_detailed"):
        cells = "".join(f"<td>{html.escape(models[m]['annotation'][key])}</td>"
                        for m in names)
        cap_rows.append(f"<tr><td class=qtype colspan=2>{key}</td>{cells}</tr>")
    qa_rows = []
    for i, p in enumerate(first["annotation"]["qa_pairs"]):
        cells = "".join(
            f"<td>{html.escape(models[m]['annotation']['qa_pairs'][i]['answer'])}</td>"
            for m in names)
        qa_rows.append(f"<tr><td class=qtype>{p.get('id', '')} {p['type']}</td>"
                       f"<td class=q>{html.escape(p['question'])}</td>{cells}</tr>")
    meta_cells = "".join(f"<td class=meta>{meta_line(models[m])}</td>" for m in names)
    qs = first.get("question_set") or {}
    if qs:
        src = f"questions by {html.escape(qs.get('model', '?'))} (set {qs.get('id', '?')})"
    else:
        src = "identical model-written questions" if len(names) > 1 else "model-written questions"
    return f"""<table>
<tr><th colspan=2>{src}</th>{head}</tr>
{''.join(cap_rows)}{''.join(qa_rows)}
<tr><td class=qtype colspan=2>meta</td>{meta_cells}</tr>
</table>
<p class=action>action: {html.escape(json.dumps(first['annotation'].get('action', {})))}</p>"""


def render_model(model: str, d: dict) -> str:
    a = d["annotation"]
    rows = []
    pairs = sorted(a["qa_pairs"], key=lambda p: QA_ORDER.index(p["type"]))
    for p in pairs:
        rows.append(f"<tr><td class=qtype>{p['type']}</td>"
                    f"<td>{html.escape(p['question'])}</td>"
                    f"<td>{html.escape(p['answer'])}</td></tr>")
    return f"""<div class=model>
<h3>{html.escape(model)}</h3>
<p class=cap><b>caption_short:</b> {html.escape(a['caption_short'])}</p>
<p class=cap><b>caption_detailed:</b> {html.escape(a['caption_detailed'])}</p>
<table><tr><th>type</th><th>question</th><th>answer</th></tr>{''.join(rows)}</table>
<p class=action>action: {html.escape(json.dumps(a.get('action', {})))}</p>
<p class=meta>{meta_line(d)}</p>
</div>"""


def render_summary(by_sample: dict) -> str:
    """Per-model totals over all samples."""
    per_model = defaultdict(list)
    for models in by_sample.values():
        for m, d in models.items():
            per_model[m].append(d)
    rows = []
    for m, ds in sorted(per_model.items()):
        n = len(ds)
        att = sum(d["meta"].get("attempts", 1) for d in ds)
        usage = [d["meta"].get("usage", {}) for d in ds]
        out_tok = sum(u.get("completion_tokens", 0) for u in usage) / n
        reas = sum(u.get("reasoning_tokens", 0) or 0 for u in usage) / n
        cost = sum(u.get("cost_usd", 0) or 0 for u in usage)
        ans_len = sum(len(p["answer"]) for d in ds for p in d["annotation"]["qa_pairs"]) / sum(
            len(d["annotation"]["qa_pairs"]) for d in ds)
        providers = sorted({d["meta"].get("provider") for d in ds if d["meta"].get("provider")})
        rows.append(f"<tr><td>{html.escape(m)}</td><td>{n}</td><td>{att - n}</td>"
                    f"<td>{out_tok:.0f}</td><td>{reas:.0f}</td><td>{ans_len:.0f}</td>"
                    f"<td>${cost:.4f}</td><td>{html.escape(', '.join(providers))}</td></tr>")
    return f"""<div class=summary><table>
<tr><th>model</th><th>samples</th><th>retries</th><th>mean out tokens</th>
<th>mean reasoning tokens</th><th>mean answer chars</th><th>total cost</th><th>provider</th></tr>
{''.join(rows)}</table></div>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("data/annotation_test"),
                    help="benchmark out_dir holding <sample>__<model>.json and frames/")
    args = ap.parse_args()
    out_dir = args.dir
    by_sample = load_results(out_dir)
    if not by_sample:
        raise SystemExit(f"no *__*.json results under {out_dir}")
    sections = []
    for sample_id, models in sorted(by_sample.items()):
        gt = next(iter(models.values()))["ground_truth"]
        figs = "".join(
            f"<figure><img src='frames/{sample_id}_{cam}.jpg' loading=lazy>"
            f"<figcaption>{cam}</figcaption></figure>"
            for cam in CAM_ORDER
            if (out_dir / "frames" / f"{sample_id}_{cam}.jpg").exists())
        wp = gt["future_waypoints_ego_frame"]
        gt_line = (f"action <b>{gt['action_label']}</b> | "
                   f"v {gt['v']:.2f} m/s, w {gt['w']:.3f} rad/s | "
                   f"past {gt.get('past_action', '?')} | traj {gt['trajectory_type']} | "
                   f"forward displacement {wp[-1][0]:.1f} m in 3 s | "
                   f"sample index {gt['sample_index']}")
        shared = len({question_key(d) for d in models.values()}) == 1
        body = (render_shared(models) if shared else
                "<div class=models>"
                + "".join(render_model(m, d) for m, d in sorted(models.items()))
                + "</div>")
        sections.append(f"""<div class=sample id='{sample_id}'>
<h2>{sample_id}</h2>
<div class=frames>{figs}</div>
<p class=gt>{gt_line}</p>
{body}
</div>""")
    page = (f"<!doctype html><meta charset=utf-8><title>annotation inspection"
            f"</title><style>{CSS}</style>\n<h1>Annotation inspection - "
            f"{len(by_sample)} samples</h1>\n{render_summary(by_sample)}\n"
            + "\n".join(sections))
    out = out_dir / "inspection.html"
    out.write_text(page)
    print(f"{out}: {len(by_sample)} samples, "
          f"{sum(len(m) for m in by_sample.values())} annotations")


if __name__ == "__main__":
    main()
