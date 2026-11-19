#!/usr/bin/env python3
"""Run offline URL-Phish preparation and fixed baseline experiments."""
import sys
if sys.version_info < (3, 11):
    raise SystemExit('Python 3.11 or newer is required. Use Python 3.11–3.13 with the pinned requirements.')
import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', nargs='+', type=Path, help='One or more original CSV files from URL-Phish version 2 only.')
    parser.add_argument('--url-column', default='url')
    parser.add_argument('--label-column', default='label')
    parser.add_argument('--prepare-only', action='store_true', help='Save the profile, features, and split without fitting models.')
    args = parser.parse_args()
    paths = args.csv or sorted((ROOT / 'data' / 'raw').rglob('*.csv'))
    if not paths:
        parser.error('No CSV files found. Download URL-Phish version 2 and place its CSV file(s) under data/raw/. See START_HERE.md.')
    if any(not p.is_file() for p in paths):
        parser.error('One or more input CSV paths do not exist.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = ROOT / 'outputs' / stamp
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / 'run.log', encoding='utf-8')], force=True)
    log = logging.getLogger('experiment')
    try:
        from src.data import prepare, create_split, save_json, sha256
        config = json.loads((ROOT / 'config.json').read_text())
        save_json(out / 'config_used.json', config)
        versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'joblib', 'tldextract']}
        save_json(out / 'environment.json', {'utc_start': stamp, 'python': sys.version,
            'platform': platform.platform(), 'machine': platform.machine(), 'packages': versions,
            'command': sys.argv, 'project_version': config['project_version']})
        locked = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], text=True, capture_output=True, check=True)
        (out / 'requirements_frozen.txt').write_text(locked.stdout, encoding='utf-8')
        code_manifest = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(ROOT.glob('src/*.py'))}
        code_manifest['run.py'] = sha256(Path(__file__))
        code_manifest['config.json'] = sha256(ROOT / 'config.json')
        save_json(out / 'code_sha256.json', code_manifest)
        log.info('Run directory: %s', out)
        log.info('Dataset: URL-Phish version 2; 0=legitimate, 1=phishing')
        log.info('Parsing URL strings locally; no websites or DNS will be contacted.')
        frame, profile = prepare(paths, out, config, args.url_column, args.label_column)
        train_idx, test_idx, split = create_split(frame, config, out)
        results = None
        if not args.prepare_only:
            from src.evaluate import train_evaluate
            results = train_evaluate(frame, train_idx, test_idx, config, out)
        from src.reporting import write_summary
        write_summary(out, profile, split, config, results)
        save_json(out / 'run_status.json', {'status': 'completed', 'mode': 'preparation_only' if args.prepare_only else 'baseline_models',
            'utc_finished': datetime.now(timezone.utc).isoformat()})
        evidence = [p for p in out.iterdir() if p.suffix in ['.json', '.txt', '.md', '.png'] or p.name in ['metrics.csv', 'feature_samples.csv', 'prediction_error_examples.csv', 'run.log']]
        log.info('Completed successfully. Evidence bundle: %s', out / 'report3_evidence.zip')
        for handler in logging.getLogger().handlers:
            handler.flush()
        with zipfile.ZipFile(out / 'report3_evidence.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for p in evidence:
                archive.write(p, arcname=p.name)
        print('\nUpload this file in our chat:\n' + str(out / 'report3_evidence.zip'), flush=True)
        print('Read SUMMARY.md and capture a Terminal screenshot of the completed run for your report.')
        return 0
    except Exception:
        logging.exception('Run stopped. No failed or incomplete run should be treated as completed research.')
        (out / 'run_status.json').write_text(json.dumps({'status': 'failed'}) + '\n')
        print('\nSend the error text or run.log so we can fix the problem. Raw inputs are unchanged.', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
