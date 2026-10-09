"""Run Apple single-frame / Samsung HDR AE-TM factorial studies."""
import argparse
import json

from .joint_experiment import run_factorial


def _integers(value):
    try:
        values = tuple(int(item) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected comma-separated integer seeds') from error
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('seeds must be nonempty and unique')
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scheme', choices=('apple', 'samsung', 'both'), default='both')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seeds', type=_integers, default=(0,))
    parser.add_argument('--noise-seeds', type=_integers, default=(0, 1))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--prepare-threads', type=int,
                        help='threads for frozen ISP preparation only; defaults to --threads')
    parser.add_argument('--candidate-chunk-size', type=int, default=4)
    parser.add_argument('--ae-lr', type=float, default=1e-3)
    parser.add_argument('--tm-lr', type=float, default=3e-4)
    parser.add_argument('--clip-risk-tolerance', type=float,
                        help='optional preview-estimated all-frame clipping excess over rule AE; default disabled')
    parser.add_argument('--weights', help='Modular Neural ISP checkpoint; defaults to shipped style 0')
    parser.add_argument('--render-ev', type=float, default=0.)
    parser.add_argument('--device', default='cpu', help='training/render device; capture preparation stays CPU')
    parser.add_argument('--evaluation-split', choices=('val', 'test'), default='test',
                        help='val selects development mode and never decodes test samples')
    args = parser.parse_args(argv)
    report = run_factorial(**vars(args))
    print(json.dumps({name: {group: result['means'] for group, result in value['groups'].items()}
                      for name, value in report['schemes'].items()}, indent=2))
    return report


if __name__ == '__main__':
    main()
