# Variant Pathogenicity Classifier (ClinVar lookup + ML)

A web app: upload a **VCF**, and every variant is classified as
**pathogenic / benign / uncertain** — by looking it up in **ClinVar** where a
curated verdict exists, and **predicting it with a machine-learning model**
where it doesn't. Results are shown as charts, a per-chromosome breakdown, and
a filterable, downloadable table.

**Tracks:** bioinformatics (variant interpretation) × data science (ML) ×
data engineering (a real SQLite database).

```
VCF upload ─▶ parse ─▶ ClinVar SQLite lookup ─┬─ found     → curated label
                                              └─ not found → ML prediction (uncertain = low confidence)
                       ─▶ chart + per-chromosome breakdown + table
```

## Files
| File | Role |
|------|------|
| `build_clinvar_db.py` | Builds `clinvar.db` (4.46M GRCh38 variants) from ClinVar — run once |
| `train_model.py` | Trains + saves `model.joblib` (shared featurizer, gene-aware eval) |
| `featurize.py` | The ONE feature function used by both training and serving (no skew) |
| `vcf_classify.py` | Core logic: VCF parse, lookup, ML fallback, bucketing |
| `app.py` | Streamlit UI: upload, charts, per-chromosome view, filterable table |
| `sample.vcf` | A tiny VCF for a quick smoke test |
| `requirements.txt` | Dependencies |

## Run locally
1. Put **`clinvar.db`** and **`model.joblib`** (built in the earlier steps) in this folder.
2. Install + launch:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. A browser tab opens. Upload `sample.vcf` to smoke-test, then a real VCF.

## Notes worth knowing
- **No train/serve skew:** the model trains on features derived from ref/alt
  (via `featurize.py`) — the exact same function the app uses on an uploaded
  variant.
- **Honesty:** ML predictions are a screening heuristic from a baseline model
  (held-out PR-AUC ≈ 0.52), not a diagnosis. The wide default "uncertain" band
  (0.30–0.70) reflects that caution and is adjustable in the UI.
- **Hosting:** `clinvar.db` is a few hundred MB — too big for free hosting as-is.
  To deploy, slim it (e.g. keep only `chrom,pos,ref,alt,label`, drop rare
  alleles, or host the DB via release asset / external store).
