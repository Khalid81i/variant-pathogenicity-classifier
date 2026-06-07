import numpy as np
import pandas as pd

VALID_CHR = [str(c) for c in range(1, 23)] + ['X', 'Y', 'MT']

def variant_type(ref, alt):
    lr, la = len(str(ref)), len(str(alt))
    if lr == 1 and la == 1:
        return 'snv'
    if lr > la:
        return 'deletion'
    if lr < la:
        return 'insertion'
    return 'mnv'

def features_frame(chrom, ref, alt):
    df = pd.DataFrame({
        'chrom': pd.Series(list(chrom)).astype(str),
        'ref':   pd.Series(list(ref)).astype(str),
        'alt':   pd.Series(list(alt)).astype(str),
    })
    df['vtype'] = [variant_type(r, a) for r, a in zip(df['ref'], df['alt'])]
    df['indel_size'] = (df['ref'].str.len() - df['alt'].str.len()).abs()
    df['log_size'] = np.log1p(df['indel_size'])
    df['chrom'] = np.where(df['chrom'].isin(VALID_CHR), df['chrom'], 'other')
    X = pd.get_dummies(df[['vtype', 'chrom']], columns=['vtype', 'chrom'])
    X['log_size'] = df['log_size'].values
    return X

def align_columns(X, columns):
    return X.reindex(columns=columns, fill_value=0)
