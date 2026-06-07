"""
build_clinvar_db.py
-------------------
Builds a slim, indexed SQLite database from ClinVar's variant_summary.txt.gz,
for fast variant lookup by genomic coordinate (GRCh38).

Reads the 419 MB file in CHUNKS so it never loads the whole thing into RAM
(this is also why it won't crash a low-memory machine).

Output: clinvar.db  with one table 'variants':
    chrom, pos, ref, alt, gene, significance (raw), label (pathogenic/benign/uncertain)

Run:
    python build_clinvar_db.py
(expects variant_summary.txt.gz in the same folder)
"""

import pandas as pd
import sqlite3

SRC = "variant_summary.txt.gz"
DB  = "clinvar.db"
COLS = ['Assembly','Chromosome','PositionVCF','ReferenceAlleleVCF',
        'AlternateAlleleVCF','GeneSymbol','ClinicalSignificance']

def to_bucket(sig):
    """Collapse ClinVar's many significance strings into 3 buckets.
    Order matters: 'Conflicting...' contains 'pathogenic' as a substring,
    so we catch it FIRST and send it to 'uncertain'."""
    s = str(sig).lower()
    if 'conflict' in s:
        return 'uncertain'
    if 'pathogenic' in s and 'benign' not in s:
        return 'pathogenic'
    if 'benign' in s:
        return 'benign'
    return 'uncertain'   # uncertain significance, drug response, risk factor, '-', etc.

def main():
    con = sqlite3.connect(DB)
    con.execute("DROP TABLE IF EXISTS variants")
    con.execute("""CREATE TABLE variants
                   (chrom TEXT, pos INTEGER, ref TEXT, alt TEXT,
                    gene TEXT, significance TEXT, label TEXT)""")

    reader = pd.read_csv(SRC, sep='\t', compression='gzip',
                         usecols=COLS, low_memory=False, chunksize=200_000)

    total = 0
    for chunk in reader:
        chunk = chunk[chunk['Assembly'] == 'GRCh38'].copy()          # GRCh38 only
        chunk['pos'] = pd.to_numeric(chunk['PositionVCF'], errors='coerce')
        chunk = chunk.dropna(subset=['pos','ReferenceAlleleVCF','AlternateAlleleVCF'])
        chunk = chunk[(chunk['ReferenceAlleleVCF'] != '-') &
                      (chunk['AlternateAlleleVCF'] != '-')]           # drop imprecise
        chunk['pos'] = chunk['pos'].astype(int)
        chunk['label'] = chunk['ClinicalSignificance'].map(to_bucket)

        rows = list(zip(chunk['Chromosome'].astype(str), chunk['pos'],
                        chunk['ReferenceAlleleVCF'], chunk['AlternateAlleleVCF'],
                        chunk['GeneSymbol'].astype(str),
                        chunk['ClinicalSignificance'].astype(str), chunk['label']))
        con.executemany("INSERT INTO variants VALUES (?,?,?,?,?,?,?)", rows)
        total += len(rows)
        print(f"  inserted {total:,} so far...")

    con.commit()
    con.execute("CREATE INDEX idx_lookup ON variants(chrom, pos, ref, alt)")
    con.commit()

    # quick summary
    print(f"\nDONE. {total:,} GRCh38 variants in {DB}")
    for label, n in con.execute(
            "SELECT label, COUNT(*) FROM variants GROUP BY label").fetchall():
        print(f"   {label:12s} {n:,}")
    con.close()

if __name__ == "__main__":
    main()
