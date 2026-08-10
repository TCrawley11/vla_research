"""Build data/annotation_test/inspection.html: key frames next to each
model's captions/QA so annotations can be eyeballed against the images.

Reads the team-schema result files written by annotate_smoke_test.py
(<sample_id>__<model>.json) plus the frames/ directory of key-frame JPEGs.

Usage: uv run python scripts/build_inspection.py
"""

import html
import json
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("data/annotation_test")
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
.qtype { white-space: nowrap; color: #9aa3b2; }
.action { font-size: .82rem; margin-top: .5rem; color: #b7e0a8; }
.meta { font-size: .72rem; color: #6f7885; margin-top: .4rem; }
"""


def load_results():
    by_sample = defaultdict(dict)
    for p in sorted(OUT_DIR.glob("*__*.json")):
        d = json.loads(p.read_text())
        by_sample[d["ground_truth"]["sample_id"]][d["model"]] = d
    return by_sample


def render_model(model: str, d: dict) -> str:
    a, meta = d["annotation"], d.get("meta", {})
    rows = []
    pairs = sorted(a["qa_pairs"], key=lambda p: QA_ORDER.index(p["type"]))
    for p in pairs:
        rows.append(f"<tr><td class=qtype>{p['type']}</td>"
                    f"<td>{html.escape(p['question'])}</td>"
                    f"<td>{html.escape(p['answer'])}</td></tr>")
    act = a.get("action", {})
    usage = meta.get("usage", {})
    return f"""<div class=model>
<h3>{html.escape(model)}</h3>
<p class=cap><b>caption_short:</b> {html.escape(a['caption_short'])}</p>
<p class=cap><b>caption_detailed:</b> {html.escape(a['caption_detailed'])}</p>
<table><tr><th>type</th><th>question</th><th>answer</th></tr>{''.join(rows)}</table>
<p class=action>action: {html.escape(json.dumps(act))}</p>
<p class=meta>attempts {meta.get('attempts', '?')},
tokens {usage.get('prompt_tokens', '?')} in / {usage.get('completion_tokens', '?')} out,
response_format {meta.get('response_format', '?')}, prompt {d.get('prompt_version', '?')}</p>
</div>"""


def main():
    by_sample = load_results()
    if not by_sample:
        raise SystemExit(f"no *__*.json results under {OUT_DIR}")
    sections = []
    for sample_id, models in sorted(by_sample.items()):
        gt = next(iter(models.values()))["ground_truth"]
        figs = "".join(
            f"<figure><img src='frames/{sample_id}_{cam}.jpg' loading=lazy>"
            f"<figcaption>{cam}</figcaption></figure>"
            for cam in CAM_ORDER
            if (OUT_DIR / "frames" / f"{sample_id}_{cam}.jpg").exists())
        wp = gt["future_waypoints_ego_frame"]
        gt_line = (f"map {gt['map_name']} | action <b>{gt['action_label']}</b> | "
                   f"v {gt['v']:.2f} m/s, w {gt['w']:.3f} rad/s | "
                   f"past {gt['his_action']} | traj {gt['trajectory_type']} | "
                   f"forward displacement {wp[-1][0]:.1f} m in 3 s | "
                   f"sample index {gt['sample_index']}")
        model_divs = "".join(render_model(m, d) for m, d in sorted(models.items()))
        sections.append(f"""<div class=sample id='{sample_id}'>
<h2>{sample_id}</h2>
<div class=frames>{figs}</div>
<p class=gt>{gt_line}</p>
<div class=models>{model_divs}</div>
</div>""")
    page = (f"<!doctype html><meta charset=utf-8><title>annotation inspection"
            f"</title><style>{CSS}</style>\n<h1>Annotation inspection - "
            f"{len(by_sample)} samples</h1>\n" + "\n".join(sections))
    out = OUT_DIR / "inspection.html"
    out.write_text(page)
    print(f"{out}: {len(by_sample)} samples, "
          f"{sum(len(m) for m in by_sample.values())} annotations")


if __name__ == "__main__":
    main()
