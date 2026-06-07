"""
vcf_classify.py
---------------
Core logic (no UI): parse a VCF, look each variant up in the ClinVar DB,
fall back to the ML model for variants not found, and bucket everything
into pathogenic / benign / uncertain.

Kept separate from the Streamlit UI so it can be unit-tested directly.
"""
import gzip
import sqlite3
import pandas as pd
from featurize import features_frame, align_columns


def normalize_chrom(c):
    """VCFs vary: 'chr1', '1', 'chrM', 'MT'. Normalise to the DB's style."""
    c = str(c).strip()
    if c.lower().startswith('chr'):
        c = c[3:]
    if c.upper() in ('M', 'MT'):
        return 'MT'
    return c.upper() if c.upper() in ('X', 'Y') else c


def parse_vcf(raw_bytes):
    """Bytes of a .vcf or .vcf.gz -> DataFrame[chrom,pos,ref,alt]."""
    if raw_bytes[:2] == b'\x1f\x8b':                      # gzip magic number
        text = gzip.decompress(raw_bytes).decode('utf-8', 'ignore')
    else:
        text = raw_bytes.decode('utf-8', 'ignore')

    rows = []
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        f = line.split('\t')
        if len(f) < 5:
            continue
        chrom, pos, _id, ref, alt = f[0], f[1], f[2], f[3], f[4]
        try:
            pos = int(pos)
        except ValueError:
            continue
        for a in alt.split(','):                          # split multi-allelic
            if a in ('.', '', '<NON_REF>'):
                continue
            rows.append((normalize_chrom(chrom), pos, ref.upper(), a.upper()))
    return pd.DataFrame(rows, columns=['chrom', 'pos', 'ref', 'alt'])


def classify(vcf_df, conn, model, columns, lo=0.30, hi=0.70):
    """For each variant: ClinVar lookup first, ML prediction if not found.
    lo/hi define the 'uncertain' band for ML predictions."""
    cur = conn.cursor()
    out = []
    for chrom, pos, ref, alt in vcf_df.itertuples(index=False):
        hit = cur.execute(
            "SELECT gene,label,significance FROM variants "
            "WHERE chrom=? AND pos=? AND ref=? AND alt=?",
            (chrom, pos, ref, alt)).fetchone()
        if hit:
            gene, label, sig = hit
            out.append((chrom, pos, ref, alt, gene, label, 'ClinVar', None))
        else:
            Xi = align_columns(features_frame([chrom], [ref], [alt]), columns)
            prob = float(model.predict_proba(Xi)[0, 1])
            label = 'pathogenic' if prob >= hi else ('benign' if prob <= lo
                                                     else 'uncertain')
            out.append((chrom, pos, ref, alt, '-', label, 'ML prediction',
                        round(prob, 3)))
    return pd.DataFrame(out, columns=['chrom', 'pos', 'ref', 'alt',
                                      'gene', 'label', 'source', 'ml_prob'])
