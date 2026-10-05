"""Build exact-stem paired manifests; target alignment is an explicit assertion.

Example (demosaiced S24 camera RGB):
python -m tm_library.prepare_data --input-dir raw_rgb --target-dir srgb \
  --metadata-dir metadata --input-encoding camera_rgb --target-aligned \
  --output train.jsonl

This builder matches exact case-sensitive stems, rejects duplicate/unmatched
files, and never independently sorts and zips directories. Use --input-layout
HWC for ambiguous NPY arrays. Metadata filenames must also match exact stems.
A reference directory is deliberately not accepted as a target source.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import ENCODINGS, IMAGE_SUFFIXES, PairedImageDataset


def _index(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f'missing directory: {directory}')
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and not path.name.startswith('.') and path.suffix.lower() in suffixes:
            if path.stem in result:
                raise ValueError(f'duplicate stem {path.stem} in {directory}')
            result[path.stem] = path.resolve()
    return result


def build_manifest(input_dir, target_dir, output, *, input_encoding,
                   metadata_dir=None, semantics_dir=None, confidence_dir=None,
                   input_layout=None, target_layout=None, semantics_layout=None,
                   scene='unknown', camera='unknown', burst_id='unknown',
                   subject_id='unknown', scenario='unknown', target_aligned=False):
    """Construct and validate all pairs before writing output.

    ``target_aligned=True`` asserts the caller has aligned the target captures;
    callers must pass it explicitly. The CLI requires --target-aligned so this
    is never guessed from filenames.
    """
    if input_encoding not in ENCODINGS:
        raise ValueError(f'unsupported input_encoding {input_encoding}')
    if target_aligned is not True:
        raise ValueError('aligned target assertion required')
    if input_encoding == 'camera_rgb' and metadata_dir is None:
        raise ValueError('camera_rgb requires metadata-dir')
    inputs = _index(Path(input_dir), IMAGE_SUFFIXES)
    targets = _index(Path(target_dir), IMAGE_SUFFIXES)
    if not inputs or set(inputs) != set(targets):
        raise ValueError(f'unmatched input/target stems: input-only {sorted(set(inputs)-set(targets))}; '
                         f'target-only {sorted(set(targets)-set(inputs))}; exact nonempty pairs required')
    indices = {}
    for field, directory in [('metadata',metadata_dir), ('semantics',semantics_dir),('confidence',confidence_dir)]:
        if directory is None:
            continue
        indices[field] = _index(Path(directory), {'.json'} if field == 'metadata' else IMAGE_SUFFIXES)
        if set(indices[field]) != set(inputs):
            raise ValueError(f'unmatched {field} stems; exact input stems required')
    output = Path(output).resolve()
    rows = []
    for stem, path in inputs.items():
        row = {'id':stem,'input':os.path.relpath(path,output.parent),
               'target':os.path.relpath(targets[stem],output.parent),
               'input_encoding':input_encoding, 'target_aligned':True,
               'scene':scene,'camera':camera,'burst_id':burst_id,
               'subject_id':subject_id,'scenario':scenario}
        for field, indexed in indices.items():
            row[field] = os.path.relpath(indexed[stem],output.parent)
        for field, layout in [('input',input_layout),('target',target_layout),('semantics',semantics_layout)]:
            if layout is not None:
                row[f'{field}_layout'] = layout
        rows.append(row)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Validate using a private sibling, so an existing destination survives errors.
    import tempfile
    is_json = output.suffix.lower() == '.json'
    payload = (json.dumps(rows, ensure_ascii=False, indent=2) + '\n' if is_json
               else ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json' if is_json else '.jsonl',dir=output.parent,delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    try:
        ds = PairedImageDataset(temp_path)
        for i in range(len(ds)):
            ds[i]
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--target-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--input-encoding',required=True,choices=sorted(ENCODINGS))
    parser.add_argument('--target-aligned',required=True,action='store_true',help='Assert targets are geometrically aligned sRGB captures')
    parser.add_argument('--metadata-dir')
    parser.add_argument('--semantics-dir')
    parser.add_argument('--confidence-dir')
    for field in ('input','target','semantics'):
        parser.add_argument(f'--{field}-layout',choices=('HWC','CHW'))
    parser.add_argument('--scene',default='unknown')
    parser.add_argument('--camera',default='unknown')
    parser.add_argument('--burst-id',default='unknown')
    parser.add_argument('--subject-id',default='unknown')
    parser.add_argument('--scenario',default='unknown')
    args = vars(parser.parse_args(argv))
    print(build_manifest(**args))


if __name__ == '__main__':
    main()
