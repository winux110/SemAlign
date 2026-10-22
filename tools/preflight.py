#!/usr/bin/env python3
"""Report missing runtime requirements without importing the model package."""
import argparse
import importlib.util
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
LOADERS = (
    'configs/_base_/datasets/dataloader/voc12.py',
    'configs/_base_/datasets/dataloader/coco_stuff.py',
    'configs/_base_/datasets/dataloader/context59.py',
)


def check(configs, check_dependencies=True):
    missing = []
    if check_dependencies:
        for name in ('torch', 'torchvision', 'mmcv', 'mmseg', 'numpy', 'einops',
                     'timm', 'ftfy', 'regex', 'matplotlib', 'scipy'):
            if importlib.util.find_spec(name) is None:
                missing.append('Python package: ' + name)
    paths = set(LOADERS)
    for config in configs:
        cfg_path = Path(config)
        if not cfg_path.is_absolute():
            cfg_path = ROOT / cfg_path
        if not cfg_path.is_file():
            missing.append('Configuration: ' + str(config))
            continue
        cfg = runpy.run_path(str(cfg_path))
        model = cfg['model']
        for key in ('pretrained', 'pretrained_text', 'load_text_embedding'):
            if model.get(key):
                paths.add(model[key])
        for split in ('train', 'val', 'test'):
            data = cfg['data'][split]
            root = Path(data.get('data_root', ''))
            paths.add(str(root))
            for key in ('img_dir', 'ann_dir', 'split'):
                if data.get(key):
                    values = data[key] if isinstance(data[key], (list, tuple)) else [data[key]]
                    for item in values:
                        value = Path(item)
                        paths.add(str(value if value.is_absolute() else root / value))
    for value in sorted(paths):
        path = Path(value)
        if not (path if path.is_absolute() else ROOT / path).exists():
            missing.append('Required path: ' + value)
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', help='Trusted local config; repeatable')
    args = parser.parse_args()
    configs = args.config or ['configs/semalign_voc_snapshot.py',
                              'configs/semalign_coco_snapshot.py']
    missing = check(configs)
    for item in missing:
        print('MISSING ' + item)
    if missing:
        print('\nPreflight FAILED.')
        return 1
    print('Dependencies and asset paths found; numerical reproduction is not verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
