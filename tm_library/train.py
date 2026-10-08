"""Train an aligned-pair experiment: python -m tm_library.train --config file.yaml."""
import argparse
from pathlib import Path
import yaml
from .engine import load_config,train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume',help='Resume an epoch-boundary last.pt checkpoint; epochs is the total desired epoch count')
    parser.add_argument('--train-manifest'); parser.add_argument('--val-manifest')
    parser.add_argument('--output-dir'); parser.add_argument('--epochs',type=int)
    parser.add_argument('--semantic-config',help='YAML external segmentation adapter configuration')
    parser.add_argument('--semantic-checkpoint',help='Override external segmentation weights')
    args=parser.parse_args()
    config=load_config(args.config)
    if args.train_manifest:
        config.setdefault('data',{})['train_manifest']=args.train_manifest
    if args.val_manifest:
        config.setdefault('data',{})['val_manifest']=args.val_manifest
    if args.output_dir:
        config['output_dir']=args.output_dir
    if args.epochs is not None:
        config.setdefault('training',{})['epochs']=args.epochs
    if args.semantic_config:
        path=Path(args.semantic_config).resolve()
        value=yaml.safe_load(path.read_text())
        value=value.get('segmentation',value)
        if value.get('weights'):
            value['weights']=str((path.parent/value['weights']).resolve())
        config.setdefault('segmentation',{}).update(value)
    if args.semantic_checkpoint:
        config.setdefault('segmentation',{})['weights']=str(Path(args.semantic_checkpoint).resolve())
    train(config,args.resume)


if __name__=='__main__':
    main()
