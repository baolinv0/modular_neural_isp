"""Import HDR scenes, generate physical RAW datasets, and validate archives."""
import argparse
import json
from pathlib import Path


def _seeds(text):
    try:
        values = tuple(int(n) for n in text.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected comma-separated nonnegative integers') from error
    if not values or min(values) < 0 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('noise seeds must be nonempty, nonnegative and unique')
    return values


def _profile(path):
    if path is None:
        return None
    from .acquisition import AcquisitionProfile
    return AcquisitionProfile.from_dict(json.loads(Path(path).expanduser().read_text(encoding='utf-8')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    importer = commands.add_parser('import', help='explicit linear source recipe to scene manifest')
    importer.add_argument('--recipe', required=True)
    importer.add_argument('--output', required=True)
    for name in ('generate', 'demo'):
        child = commands.add_parser(name)
        child.add_argument('--output', required=True)
        child.add_argument('--scheme', choices=('apple', 'samsung', 'both'), default='both')
        child.add_argument('--acquisition-profile')
        child.add_argument('--noise-seeds', type=_seeds, default=(0, 1))
        child.add_argument('--seed', type=int, default=0)
        child.add_argument('--threads', type=int, default=1)
        child.add_argument('--render-ev', type=float, default=0.)
        child.add_argument('--no-noise', action='store_true', help='diagnostic capture only; previews remain noisy')
        if name == 'generate':
            child.add_argument('--manifest', required=True)
        else:
            child.add_argument('--scenes', type=int, default=12)
            child.add_argument('--size', type=int, default=32)
    validator = commands.add_parser('validate')
    validator.add_argument('--manifest', required=True)
    validator.add_argument('--output', help='optional validation JSON path')
    validator.add_argument('--threads', type=int, default=1)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    if command == 'import':
        from .pipeline_sources import import_source_recipe
        result = import_source_recipe(args['recipe'], args['output'])
    elif command == 'validate':
        from .pipeline import validate_acquisition_dataset
        report = validate_acquisition_dataset(args.pop('manifest'), **args)
        print(json.dumps(report, indent=2))
        return report
    else:
        from .pipeline import generate_acquisition_dataset
        args['acquisition'] = _profile(args.pop('acquisition_profile'))
        args['noisy'] = not args.pop('no_noise')
        if command == 'demo':
            from .joint_data import write_joint_dataset
            directory = Path(args.pop('output')).expanduser().resolve()
            if directory.exists():
                raise FileExistsError('demo output already exists; choose a new directory')
            args['manifest'] = write_joint_dataset(directory / 'sources', scenes=args.pop('scenes'),
                                                  size=args.pop('size'), seed=args['seed'])
            args['output'] = directory / 'acquisition'
        result = generate_acquisition_dataset(**args)
    print(result)
    return result


if __name__ == '__main__':
    main()
