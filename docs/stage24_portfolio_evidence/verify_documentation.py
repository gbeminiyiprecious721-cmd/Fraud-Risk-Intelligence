"""Check Stage 24 Markdown targets and published values against saved evidence.

Run from the repository root with the existing research/test environment:
    .venv/bin/python docs/stage24_portfolio_evidence/verify_documentation.py
No historical dataset, prediction file, model deserialization or fitting occurs.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

import mistune

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "README.md", "docs/ARCHITECTURE.md", "docs/PORTFOLIO_WALKTHROUGH.md",
    "docs/PORTFOLIO_CASE_STUDY.md", "docs/STAGE24_PORTFOLIO_DOCUMENTATION.md",
    "docs/ML_WORKFLOW_STATE.md",
]
parse = mistune.create_markdown(renderer="ast", plugins=["table"])


def walk(nodes):
    for node in nodes:
        yield node
        yield from walk(node.get("children", []))


def plain(node):
    return node.get("raw", "") + "".join(plain(c) for c in node.get("children", []))


def anchors(path):
    seen, result = {}, set()
    for n in walk(parse(path.read_text())):
        if n["type"] != "heading":
            continue
        slug = re.sub(r"\s", "-", "".join(
            c for c in plain(n).lower()
            if c in " -_" or unicodedata.category(c)[0] in "LN"
        ))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        result.add(slug if not count else f"{slug}-{count}")
    return result


def main():
    errors, links, images, diagrams = [], 0, 0, []
    for name in FILES:
        path = ROOT / name
        text = path.read_text()
        assert text.count("```") % 2 == 0, f"Unclosed fence: {name}"
        for node in walk(parse(text)):
            if node["type"] == "block_code" and node.get("attrs", {}).get("info") == "mermaid":
                diagrams.append({"file": name, "source": node["raw"]})
            if node["type"] not in ("link", "image"):
                continue
            url = urlsplit(node["attrs"]["url"])
            if url.scheme or url.netloc:
                continue
            links += 1
            images += node["type"] == "image"
            target = path.parent / unquote(url.path) if url.path else path
            if not target.exists():
                errors.append(f"{name}: missing {node['attrs']['url']}")
            elif url.fragment and target.suffix == ".md" and unquote(url.fragment) not in anchors(target):
                errors.append(f"{name}: missing anchor {node['attrs']['url']}")
    assert len(diagrams) == 3, len(diagrams)
    assert not errors, errors

    read = lambda p: json.loads((ROOT / p).read_text())
    manifest = read("artifacts/frozen_rf_fraud_risk_v1.joblib.manifest.json")
    digest = hashlib.sha256((ROOT / "artifacts/frozen_rf_fraud_risk_v1.joblib").read_bytes()).hexdigest()
    assert digest == manifest["artifact_sha256"] == "2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04"
    assert len(manifest["raw_feature_schema"]) == 392
    assert len(manifest["transformed_feature_schema"]) == 813
    assert manifest["training_partition"]["n_rows"] == 438125
    # Read registry summaries only, never the evaluation's rows or predictions.
    records = [json.loads(line) for line in (ROOT / "docs/experiment_registry.jsonl").read_text().splitlines()]
    final = [r["metrics"] for r in records if r.get("workflow_stage") == 10
             and (r.get("metrics") or {}).get("n_rows") == 76855]
    assert len(final) == 1
    f = final[0]
    assert (f["n_positive"], f["average_precision"], f["roc_auc"]) == (2682, .42784, .86637)
    assert (f["k_frac_0.01"]["precision_at_k"], f["k_frac_0.01"]["recall_at_k"],
            f["k_frac_0.01"]["k"], f["k_frac_0.01"]["tp_at_k"]) == (.8583, .2461, 769, 660)
    b = read("ui/benchmark-v1.json")
    assert (b["rows"], b["positives"]) == (75560, 2617)
    expected = {"Random Forest": (.44700, .87333), "LightGBM": (.54590, .90937),
                "XGBoost": (.55178, .91044), "CatBoost": (.49048, .87754)}
    readme = (ROOT / "README.md").read_text()
    published = {name: (float(ap), float(roc)) for name, ap, roc in re.findall(
        r"\| (Random Forest|LightGBM|XGBoost|CatBoost) \| (0\.\d+) \| (0\.\d+) \|", readme
    )}
    assert published == expected, published
    for key in ("average_precision", "roc_auc"):
        assert f"**{f[key]:.5f}**" in readme
    for key in ("precision_at_k", "recall_at_k"):
        assert f"**{f['k_frac_0.01'][key]:.4f}**" in readme
    for model in b["models"]:
        metric = model["metrics"]
        assert tuple(round(metric[k], 5) for k in ("average_precision", "roc_auc")) == expected[model["name"]]
    assert sum(len(m["metrics"]["points"]) for m in b["models"]) == 804
    assert (b["bootstrap"]["draws"], b["bootstrap"]["block_count"]) == (1000, 20)
    c = read("ui/capacity-v1.json")
    assert (c["n"], c["positives"], c["checkpoint_count"]) == (75560, 2617, 201)
    assert (b["models"][0]["metrics"]["top1"]["k"], b["models"][0]["metrics"]["top1"]["tp"]) == (756, 618)
    m = read("ui/monitoring-v1.json")
    assert [w["n"] for w in m["windows"]] == [25186, 25187, 25187]
    assert all(a["end"] < z["start"] for a, z in zip(m["windows"], m["windows"][1:]))
    g = read("explainability_assets/global-v1.json")
    l = read("explainability_assets/local-v1.json")
    assert len(g["groups"]) == 392 and len(l["records"]) == 240
    assert g["population"]["n"] == 512 and g["population"]["labels_used"] is False
    # Independently count the source adapter's literal public cohort bounds.
    import ast
    module = ast.parse((ROOT / "src/render_demo.py").read_text())
    constants = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("PUBLIC_DASHBOARD_BATCH_SIZE", "PUBLIC_DASHBOARD_SEED_ALLOWLIST"):
                    constants[target.id] = ast.literal_eval(node.value)
    assert constants["PUBLIC_DASHBOARD_BATCH_SIZE"] * len(constants["PUBLIC_DASHBOARD_SEED_ALLOWLIST"]) == 120
    report = {
        "scope": FILES, "internal_links_checked": links, "inline_images_checked": images,
        "errors": errors, "mermaid_diagrams": [{"file": d["file"], "sha256": hashlib.sha256(d["source"].encode()).hexdigest()} for d in diagrams],
        "numerical_sources": ["docs/experiment_registry.jsonl", "artifacts/frozen_rf_fraud_risk_v1.joblib.manifest.json", "ui/benchmark-v1.json", "ui/capacity-v1.json", "ui/monitoring-v1.json", "explainability_assets/global-v1.json", "explainability_assets/local-v1.json"],
        "frozen_sha256": digest, "closed_data_access": False,
        "note": "Published summaries and versioned assets only. Mermaid rendering is checked separately in Chrome.",
    }
    out = ROOT / "docs/stage24_portfolio_evidence/documentation-checks.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
