"""Train an aligned-pair experiment: python -m tm_library.train --config file.yaml."""
import argparse
from .engine import load_config,train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume',help='Resume an epoch-boundary last.pt checkpoint; epochs is the total desired epoch count')
    parser.add_argument('--train-manifest'); parser.add_argument('--val-manifest')
    parser.add_argument('--output-dir'); parser.add_argument('--epochs',type=int)
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
    train(config,args.resume)


if __name__=='__main__':
    main()
