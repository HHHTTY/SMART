import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path('/hpc2hdd/home/aimslab/ChengtangZhan')
REPO = ROOT / 'ttt/ibm复现/official-code'
CKPT_DIR = ROOT / 'ttt/ibm复现/iclr'
# The default is retained for backwards-compatible smoke runs.  ZIP-derived
# evaluations override this with SDBS_TOKENIZED_DATA_DIR so the source and
# target are read from the newly materialized archive view.
DATA_DIR = Path(os.environ.get(
    'SDBS_TOKENIZED_DATA_DIR',
    str(ROOT / 'Dataset/sdbs_tokenized_datasets/SDBS_final'),
))
SRC_ALL = DATA_DIR / 'src.txt'
TGT_ALL = DATA_DIR / 'tgt.txt'
RUN = Path(sys.argv[1])
MS_MARKER = os.environ.get('SDBS_MS_MARKER', 'E0Pos')
if MS_MARKER not in {'E0Pos', 'EI75eV', 'none'}:
    raise ValueError(f'unsupported SDBS_MS_MARKER={MS_MARKER!r}')

MODELS = {
    'nmr': {'h': True, 'c': True, 'ir': False, 'ms': False},
    'ir': {'h': False, 'c': False, 'ir': True, 'ms': False},
    'ms': {'h': False, 'c': False, 'ir': False, 'ms': True},
    'ir_ms': {'h': False, 'c': False, 'ir': True, 'ms': True},
    'nmr_ir': {'h': True, 'c': True, 'ir': True, 'ms': False},
    'nmr_ms': {'h': True, 'c': True, 'ir': False, 'ms': True},
    'all': {'h': True, 'c': True, 'ir': True, 'ms': True},
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_metrics(rows):
    fields = [
        'model', 'status', 'test_rows', 'pred_lines', 'top1', 'top5', 'top10',
        'source_max_tokens', 'source_over_192', 'inference_sec',
        'evaluation_sec', 'error',
    ]
    with (RUN / 'metrics_final.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fields})
    write_json(RUN / 'metrics_final.json', rows)


def split_prepared_line(line):
    """Split an existing SDBS OpenNMT line without re-tokenizing any value."""
    nmr_start = line.index(' 1HNMR ')
    ir_start = line.index(' IR ')
    ms_start = line.index(' MS ', ir_start)
    formula = line[:nmr_start]
    nmr = line[nmr_start:ir_start].strip()
    ir = line[ir_start:ms_start]
    # Keep the prepared peak pairs unchanged; only the channel marker is varied.
    ms = line[ms_start:]
    if MS_MARKER == 'none':
        ms = ms.replace(' MS ', ' ', 1)
    else:
        ms = ms.replace(' MS ', f' {MS_MARKER} ', 1)
    return formula, nmr, ir, ms


def compose(formula, nmr, ir, ms, modes):
    parts = [formula]
    if modes['h'] or modes['c']:
        parts.append(nmr)
    if modes['ir']:
        parts.append(ir)
    if modes['ms']:
        parts.append(ms)
    return ' '.join(parts).strip()


def main():
    started = datetime.now().astimezone().isoformat(timespec='seconds')
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'logs').mkdir(parents=True, exist_ok=True)
    source_lines = SRC_ALL.read_text(encoding='utf-8').splitlines()
    target_lines = TGT_ALL.read_text(encoding='utf-8').splitlines()
    if len(source_lines) != 3669 or len(target_lines) != 3669:
        raise RuntimeError(f'expected 3669 prepared rows; got src={len(source_lines)} tgt={len(target_lines)}')
    if any(' IR ' not in line or ' MS ' not in line for line in source_lines):
        raise RuntimeError('prepared SDBS source does not contain the expected IR/MS markers')

    prepared = [split_prepared_line(line) for line in source_lines]
    target_path = RUN / 'target-test.txt'
    target_path.write_text(''.join(line + '\n' for line in target_lines), encoding='utf-8')

    src_stats = {}
    for model, modes in MODELS.items():
        lines = [compose(formula, nmr, ir, ms, modes) for formula, nmr, ir, ms in prepared]
        src_path = RUN / model / 'src-test.txt'
        src_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_text(''.join(line + '\n' for line in lines), encoding='utf-8')
        lengths = [len(line.split()) for line in lines]
        src_stats[model] = {
            'input_modes': modes,
            'rows': len(lines),
            'source_max_tokens': max(lengths),
            'source_over_192': sum(length > 192 for length in lengths),
            'source_file': str(src_path),
        }

    manifest = {
        'status': 'running',
        'purpose': 'full SDBS external-set evaluation of the seven ICLR OpenNMT checkpoints using ZIP-derived tokenized lines',
        'started_at': started,
        'dataset_dir': str(DATA_DIR),
        'dataset_schema': 'SDBS_final.v1',
        'dataset_rows': 3669,
        'dataset_unique_inchikeys': 3669,
        'prepared_source': {
            'src': str(SRC_ALL),
            'src_sha256': sha256(SRC_ALL),
            'tgt': str(TGT_ALL),
            'tgt_sha256': sha256(TGT_ALL),
            'policy': 'read the 3669-row source/target files generated directly from SDBS.zip; no parquet fallback',
            'representation': 'formula + 1HNMR + 13CNMR + 400-point IR + SDBS EI 75 eV',
        },
        'paper_source_exact_canonical_smiles_overlap': {
            'rows': 1449,
            'fraction': 1449 / 3669,
            'comparison': 'exact canonical_smiles strings against all 245 original paper parquet files (789345 unique smiles)',
            'interpretation': 'external SDBS set, not a strict molecule-disjoint zero-shot split',
        },
        'checkpoint_source': str(CKPT_DIR),
        'beam_size': 10,
        'n_best': 10,
        'min_length': 5,
        'gpu': 0,
        'input_policy': {
            'formula_nmr_ir': 'copied from the prepared SDBS src.txt segments exactly',
            'ms': f'prepared MS EI peak pairs copied exactly; marker MS is represented as {MS_MARKER}',
            'ei75ev_tag': 'omitted; no EI75eV token is added',
            'ms_caveat': 'EI 75 eV is not equivalent to the checkpoint training distribution of positive/negative multi-energy MS/MS',
        },
        'source_stats': src_stats,
        'results': [],
    }
    write_json(RUN / 'manifest.json', manifest)

    results = []
    translator_code = (
        'import argparse,sys,torch.serialization; '
        'torch.serialization.add_safe_globals([argparse.Namespace]); '
        'from onmt.bin.translate import main; '
        "sys.argv=['onmt_translate']+sys.argv[1:]; main()"
    )
    evaluator = REPO / 'benchmark/analyse_results.py'

    for model in MODELS:
        checkpoint = CKPT_DIR / f'{model}.pt'
        model_dir = RUN / model
        pred = model_dir / 'pred-test.txt'
        translate_log = RUN / 'logs' / f'{model}_translate.log'
        eval_log = RUN / 'logs' / f'{model}_eval.log'
        result = {
            'model': f'{model}.pt',
            'status': 'running',
            'test_rows': 3669,
            'pred_lines': 0,
            'source_max_tokens': src_stats[model]['source_max_tokens'],
            'source_over_192': src_stats[model]['source_over_192'],
            'error': '',
        }
        t0 = time.time()
        cmd = [
            sys.executable, '-c', translator_code,
            '-model', str(checkpoint),
            '-src', str(model_dir / 'src-test.txt'),
            '-output', str(pred),
            '-beam_size', '10', '-n_best', '10', '-min_length', '5', '-gpu', '0',
        ]
        with translate_log.open('w', encoding='utf-8') as log:
            proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        result['inference_sec'] = round(time.time() - t0, 1)
        if proc.returncode != 0:
            result['status'] = 'inference_failed'
            result['error'] = f'translator_exit_{proc.returncode}'
        else:
            result['pred_lines'] = sum(1 for _ in pred.open(encoding='utf-8'))
            if result['pred_lines'] != 36690:
                result['status'] = 'bad_prediction_count'
                result['error'] = f'expected_36690_lines_got_{result["pred_lines"]}'
            else:
                t1 = time.time()
                evaluation = subprocess.run(
                    [sys.executable, str(evaluator), '--pred_path', str(pred), '--test_path', str(target_path)],
                    cwd=REPO, capture_output=True, text=True, env=os.environ.copy(),
                )
                eval_log.write_text(evaluation.stdout + evaluation.stderr, encoding='utf-8')
                result['evaluation_sec'] = round(time.time() - t1, 1)
                if evaluation.returncode != 0:
                    result['status'] = 'evaluation_failed'
                    result['error'] = f'evaluator_exit_{evaluation.returncode}'
                else:
                    for rank, value in re.findall(r'Top\s+(\d+):\s+([0-9.]+)', evaluation.stdout):
                        if rank in {'1', '5', '10'}:
                            result[f'top{rank}'] = float(value)
                    result['status'] = 'ok' if all(key in result for key in ('top1', 'top5', 'top10')) else 'metrics_missing'
                    if result['status'] != 'ok':
                        result['error'] = 'expected_Top_1_5_10_not_found'
        results.append(result)
        manifest['results'] = results
        write_json(RUN / 'manifest.json', manifest)
        write_metrics(results)
        print(
            f'RESULT {model}: status={result["status"]} top1={result.get("top1")} '
            f'top5={result.get("top5")} top10={result.get("top10")} '
            f'pred_lines={result["pred_lines"]} sec={result["inference_sec"]}',
            flush=True,
        )

    manifest['status'] = 'complete' if all(row['status'] == 'ok' for row in results) else 'partial'
    manifest['completed_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    write_json(RUN / 'manifest.json', manifest)
    write_json(RUN / 'manifest_final.json', manifest)
    print('\nFINAL SUMMARY', flush=True)
    for row in results:
        print(row, flush=True)
    print(f'ARTIFACTS {RUN}', flush=True)


if __name__ == '__main__':
    main()
