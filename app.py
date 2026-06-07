import sqlite3
import joblib
import pandas as pd
import altair as alt
import streamlit as st
from vcf_classify import parse_vcf, classify
import os, urllib.request

DB_URL = "https://github.com/Khalid81i/variant-pathogenicity-classifier/releases/download/v1.0/clinvar.db"
MAX_VARIANTS = 5000

st.set_page_config(page_title="Variant Pathogenicity Classifier",
                   page_icon="🧬", layout="wide")

CHROM_ORDER = [str(c) for c in range(1, 23)] + ['X', 'Y', 'MT']
LABELS = ['pathogenic', 'uncertain', 'benign']
COLORS = {'pathogenic': '#B23A48', 'uncertain': '#C77D2E', 'benign': '#2E7D5B'}

st.markdown("""
<style>
  .block-container{padding-top:2.2rem; max-width:1150px}
  h1,h2,h3{font-family:Georgia,'Times New Roman',serif; letter-spacing:-.01em}
  [data-testid="stMetricValue"]{font-size:1.7rem}
  .legend-row span{display:inline-block; margin-right:18px; font-size:.85rem; color:#5C626B}
  .swatch{display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:6px; vertical-align:middle}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    b = joblib.load("model.joblib")
    return b['model'], b['columns']

@st.cache_resource
def get_conn():
    if not os.path.exists("clinvar.db"):
        with st.spinner("First-time setup: downloading the ClinVar database (~330 MB)\u2026"):
            urllib.request.urlretrieve(DB_URL, "clinvar.db")
    return sqlite3.connect("clinvar.db", check_same_thread=False)

def color_scale():
    return alt.Scale(domain=LABELS, range=[COLORS[l] for l in LABELS])

st.title("Variant Pathogenicity Classifier")
st.markdown(
    "Upload a **VCF** file. Each variant is resolved against **ClinVar** "
    "(4.46M curated GRCh38 variants); any not found are scored by a "
    "**machine-learning model**, then bucketed as pathogenic, benign or uncertain.")
st.markdown(
    '<div class="legend-row">'
    '<span><span class="swatch" style="background:#B23A48"></span>Pathogenic</span>'
    '<span><span class="swatch" style="background:#C77D2E"></span>Uncertain</span>'
    '<span><span class="swatch" style="background:#2E7D5B"></span>Benign</span>'
    '</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    lo, hi = st.slider(
        "ML 'uncertain' band (probability of pathogenic)",
        0.0, 1.0, (0.30, 0.70), 0.05,
        help="Below the lower value = benign, above the upper = pathogenic, between = uncertain.")
    st.caption(f"benign \u2264 {lo:.2f}  \u00b7  uncertain  \u00b7  pathogenic \u2265 {hi:.2f}")
    st.markdown("---")
    st.caption("ML predictions are a screening heuristic from a baseline model "
               "(held-out PR-AUC \u2248 0.52) \u2014 not a clinical diagnosis.")

try:
    model, columns = load_model()
    conn = get_conn()
except Exception as e:
    st.error(f"Could not load clinvar.db / model.joblib:\n\n{e}")
    st.stop()

up = st.file_uploader("Upload a VCF file (.vcf or .vcf.gz)", type=["vcf", "gz"])
if up is None:
    st.info("Waiting for a VCF upload. Each row's CHROM, POS, REF and ALT are used.")
    st.stop()

vcf_df = parse_vcf(up.getvalue())
if vcf_df.empty:
    st.error("No variants parsed. Is this a valid VCF (tab-separated, with data rows)?")
    st.stop()

truncated = len(vcf_df) > MAX_VARIANTS
if truncated:
    vcf_df = vcf_df.head(MAX_VARIANTS)

with st.spinner(f"Classifying {len(vcf_df):,} variants..."):
    res = classify(vcf_df, conn, model, columns, lo=lo, hi=hi)
if truncated:
    st.warning(f"Large file \u2014 showing the first {MAX_VARIANTS:,} variants (demo cap).")

st.markdown("### Summary")
c = res['label'].value_counts()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Variants", f"{len(res):,}")
m2.metric("Pathogenic", f"{c.get('pathogenic', 0):,}")
m3.metric("Benign", f"{c.get('benign', 0):,}")
m4.metric("Uncertain", f"{c.get('uncertain', 0):,}")
m5.metric("Resolved by ClinVar", f"{(res['source'] == 'ClinVar').sum():,} / {len(res):,}")

st.markdown("")
left, right = st.columns([1, 1.4])

with left:
    st.markdown("**Classification**")
    a = res.groupby('label').size().reset_index(name='count')
    chart_a = (alt.Chart(a).mark_bar(cornerRadius=2)
               .encode(
                   x=alt.X('label:N', sort=LABELS, title=None, axis=alt.Axis(labelAngle=0)),
                   y=alt.Y('count:Q', title='Variants'),
                   color=alt.Color('label:N', scale=color_scale(), legend=None),
                   tooltip=['label', 'count'])
               .properties(height=300))
    st.altair_chart(chart_a, use_container_width=True)

with right:
    st.markdown("**By chromosome**")
    b = res.groupby(['chrom', 'label']).size().reset_index(name='count')
    chart_b = (alt.Chart(b).mark_bar()
               .encode(
                   x=alt.X('chrom:N', sort=CHROM_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
                   y=alt.Y('count:Q', stack='zero', title='Variants'),
                   color=alt.Color('label:N', scale=color_scale(), sort=LABELS, title='Class'),
                   tooltip=['chrom', 'label', 'count'])
               .properties(height=300))
    st.altair_chart(chart_b, use_container_width=True)

st.markdown("### All variants")
f1, f2 = st.columns(2)
labels_sel = f1.multiselect("Filter by classification", LABELS, default=LABELS)
src_sel = f2.multiselect("Filter by source", ['ClinVar', 'ML prediction'],
                         default=['ClinVar', 'ML prediction'])
view = res[res['label'].isin(labels_sel) & res['source'].isin(src_sel)]

def style_label(v):
    return f"color:{COLORS.get(v, '#15181D')}; font-weight:600"

styled = view.style.map(style_label, subset=['label'])
st.dataframe(styled, use_container_width=True, hide_index=True)
st.download_button("Download results as CSV",
                   view.to_csv(index=False).encode(),
                   "variant_classifications.csv", "text/csv")
