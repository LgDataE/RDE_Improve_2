import argparse
import os
import os.path as op
import sys

if __file__:
    repo_root = op.abspath(op.join(op.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from utils.iotools import read_json, write_json


def _infer_split(anno: dict):
    split = anno.get('split', None)
    if split is not None:
        return split
    fp = str(anno.get('file_path', '')).lower()
    if fp.startswith('train') or 'train_query' in fp:
        return 'train'
    if fp.startswith('test') or 'test_query' in fp:
        return 'test'
    if fp.startswith('val') or fp.startswith('valid') or fp.startswith('validation'):
        return 'val'
    return None


def _pick_anno_path(dataset_dir: str):
    candidates = [
        'reid_raw_clean.json',
        'reid_raw.json',
        'caption_all_clean.json',
        'caption_all.json',
    ]
    for name in candidates:
        p = op.join(dataset_dir, name)
        if op.exists(p):
            return p
    raise RuntimeError(
        f"No CUHK-PEDES annotation json found under '{dataset_dir}'. Expected one of: "
        + ', '.join(candidates)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True, type=str)
    parser.add_argument('--dataset_dirname', default='CUHK-PEDES', type=str)
    parser.add_argument('--output_name', default='', type=str)
    parser.add_argument('--write_clean', action='store_true', default=True)
    args = parser.parse_args()

    dataset_dir = op.join(args.root_dir, args.dataset_dirname)
    if (not op.exists(op.join(dataset_dir, 'imgs'))) and op.exists(op.join(dataset_dir, args.dataset_dirname, 'imgs')):
        dataset_dir = op.join(dataset_dir, args.dataset_dirname)

    img_dir = op.join(dataset_dir, 'imgs')
    if not op.exists(img_dir):
        raise RuntimeError(f"'{img_dir}' is not available")

    anno_path = _pick_anno_path(dataset_dir)
    annos = read_json(anno_path)
    if not isinstance(annos, list):
        raise RuntimeError(f"Expected list json at '{anno_path}', but got {type(annos)}")

    if args.output_name:
        out_path = op.join(dataset_dir, args.output_name)
    else:
        base = op.basename(anno_path)
        if base.endswith('.json'):
            base = base[:-5]
        out_path = op.join(dataset_dir, f"{base}_clean.json")

    kept = []
    missing = 0
    no_split = 0

    for a in annos:
        fp = a.get('file_path', None)
        if not fp:
            missing += 1
            continue

        full_img_path = op.join(img_dir, fp)
        if not op.exists(full_img_path):
            missing += 1
            continue

        split = _infer_split(a)
        if split is None:
            no_split += 1
            continue

        a2 = dict(a)
        a2['split'] = split
        kept.append(a2)

    stats = {
        'input_anno_path': anno_path,
        'output_anno_path': out_path,
        'total': len(annos),
        'kept': len(kept),
        'missing_images_or_file_path': missing,
        'missing_split_and_cannot_infer': no_split,
    }
    print(stats)

    if args.write_clean:
        write_json(kept, out_path)


if __name__ == '__main__':
    main()
