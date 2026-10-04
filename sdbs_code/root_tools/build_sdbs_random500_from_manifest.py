#!/usr/bin/env python3
"""Select a fixed 500-row subset from the saved 1000-row manifest and build inputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from build_sdbs_random1000_combinations import COMBINATIONS, make_record

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--manifest1000', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--manifest500', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20260923)
    args = ap.parse_args()
    cols = ['global_row_id','sdbs_no','smiles','canonical_smiles','h_nmr_raw_json','c_nmr_raw_json','ms_raw_json','ir_1800']
    frame = pd.read_parquet(args.input, columns=cols).set_index('global_row_id', drop=False)
    m1000 = json.loads(args.manifest1000.read_text())
    ids = m1000['global_row_ids']
    selected = frame.loc[ids].sample(n=500, random_state=args.seed).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'source_manifest': str(args.manifest1000), 'input': str(args.input), 'sample_size': 500, 'seed': args.seed,
                'global_row_ids': [int(x) for x in selected.global_row_id], 'sdbs_no': [str(x) for x in selected.sdbs_no]}
    args.manifest500.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for name, modes in COMBINATIONS.items():
        path = args.output_dir / f'{name}_test500.jsonl'
        with path.open('w', encoding='utf-8') as h:
            for _, row in selected.iterrows():
                h.write(json.dumps(make_record(row, modes), ensure_ascii=False) + '\n')
        print(name, path, len(selected))

if __name__ == '__main__':
    main()
