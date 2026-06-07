"""
app.py  -- Variant Pathogenicity Classifier
Upload a VCF -> look each variant up in ClinVar -> predict the rest with the
ML model -> visualise the breakdown.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
(expects clinvar.db and model.joblib in the same folder)
"""
import sqlite3
import joblib
import pandas as pd
import streamlit as st
from vcf_classify import parse_vcf, classify

st.set_page_config(page_title="Variant Pathogenicity Classifier", layout="wide")

CHROM_ORDER = [str(c) for c in range(1, 23)] + ['X', 'Y', 'MT']
LABEL_COLORS = {'pathogenic': '#C0392B', 'benign': '#27AE60', 'uncertain': '#E67E22'}


@st.cache_resource
def load_model():
    b = joblib.load("model.joblib")
    return b['model'], b['columns']


@st.cache_resource
def get_conn():
    return sqlite3.connect("clinvar.db", check_same_thread=False)


# ---------------- header ----------------
st.title("🧬 Variant Pathogenicity Classifier")
st.markdown(
    "Upload a **VCF** file. Each variant is looked up in **ClinVar** "
    "(4.46M curated GRCh38 variants); any not found are scored by a "
    "**machine-learning model**. Results are bucketed into "
    "**pathogenic / benign / uncertain**."
)
st.warning(
    "ML predictions are a rough screening heuristic from a baseline model "
    "(held-out PR-AUC ≈ 0.52), **not a clinical diagnosis**. ClinVar lookups "
    "reflect curated submissions and can change over time."
)

# ---------------- sidebar controls ----------------
with st.sidebar:
    st.header("Settings")
    lo, hi = st.slider(
        "ML 'uncertain' band (probability of pathogenic)",
        0.0, 1.0, (0.30, 0.70), 0.05,
        help="Predicted probability below the lower value = benign, above the "
             "upper value = pathogenic, in between = uncertain.")
    st.caption(f"benign ≤ {lo:.2f}  ·  uncertain  ·  pathogenic ≥ {hi:.2f}")

try:
    model, columns = load_model()
    conn = get_conn()
except Exception as e:
    st.error(f"Could not load clinvar.db / model.joblib — are they in this folder?\n\n{e}")
    st.stop()

# ---------------- upload + classify ----------------
up = st.file_uploader("Upload a VCF file (.vcf or .vcf.gz)", type=["vcf", "gz"])
if up is None:
    st.info("Waiting for a VCF upload. Each line's CHROM, POS, REF and ALT are used.")
    st.stop()

vcf_df = parse_vcf(up.getvalue())
if vcf_df.empty:
    st.error("No variants parsed. Is this a valid VCF (tab-separated, with data rows)?")
    st.stop()

with st.spinner(f"Classifying {len(vcf_df):,} variants..."):
    res = classify(vcf_df, conn, model, columns, lo=lo, hi=hi)

# ---------------- summary metrics ----------------
st.subheader("Summary")
c = res['label'].value_counts()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Variants", f"{len(res):,}")
m2.metric("Pathogenic", f"{c.get('pathogenic', 0):,}")
m3.metric("Benign", f"{c.get('benign', 0):,}")
m4.metric("Uncertain", f"{c.get('uncertain', 0):,}")
m5.metric("From ClinVar", f"{(res['source'] == 'ClinVar').sum():,} / {len(res):,}")

# ---------------- charts ----------------
left, right = st.columns(2)

with left:
    st.markdown("**Classification (by source)**")
    by_src = (res.groupby(['label', 'source']).size()
              .unstack(fill_value=0)
              .reindex(['pathogenic', 'uncertain', 'benign']))
    st.bar_chart(by_src)

with right:
    st.markdown("**Per-chromosome breakdown**")
    by_chr = (res.groupby(['chrom', 'label']).size().unstack(fill_value=0))
    by_chr = by_chr.reindex([c for c in CHROM_ORDER if c in by_chr.index])
    for lab in ['pathogenic', 'benign', 'uncertain']:
        if lab not in by_chr.columns:
            by_chr[lab] = 0
    st.bar_chart(by_chr[['pathogenic', 'benign', 'uncertain']])

# ---------------- filterable table ----------------
st.subheader("All variants")
f1, f2 = st.columns(2)
labels_sel = f1.multiselect("Filter by classification",
                            ['pathogenic', 'benign', 'uncertain'],
                            default=['pathogenic', 'benign', 'uncertain'])
src_sel = f2.multiselect("Filter by source", ['ClinVar', 'ML prediction'],
                         default=['ClinVar', 'ML prediction'])

view = res[res['label'].isin(labels_sel) & res['source'].isin(src_sel)]
st.dataframe(view, use_container_width=True, hide_index=True)
st.download_button("Download results as CSV",
                   view.to_csv(index=False).encode(),
                   "variant_classifications.csv", "text/csv")
