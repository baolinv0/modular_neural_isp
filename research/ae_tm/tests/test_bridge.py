import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'

def module(name):
    path = TOOLS / f'{name}.py'
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

bridge = module('bridge')
preflight = module('preflight')

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(bridge, 'bridge implementation is not present yet')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def task(self, code='print("ok")', **kwargs):
        value = {'id': 'fixture', 'timeout_seconds': 2,
                 'steps': [{'name': 'probe', 'argv': ['{python}', '-c', code]}]}
        value.update(kwargs)
        return value

    def test_success_records_exit_code_and_no_research_claim(self):
        result = bridge.run_task(self.task(), self.root, self.root / 'run')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['steps'][0]['returncode'], 0)
        self.assertFalse(result['scientific_claim_verified'])
        self.assertTrue((self.root / 'run' / 'task.json').exists())

    def test_failed_step_prevents_next_step(self):
        task = self.task('raise SystemExit(7)')
        task['steps'].append({'name': 'never', 'argv': ['{python}', '-c', 'print("never")']})
        result = bridge.run_task(task, self.root, self.root / 'run')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['steps']), 1)
        self.assertEqual(result['steps'][0]['returncode'], 7)

    def test_timeout_is_not_success(self):
        result = bridge.run_task(self.task('import time; time.sleep(4)', timeout_seconds=0.15),
                                 self.root, self.root / 'run')
        self.assertEqual(result['status'], 'timed_out')

    def test_existing_output_refused(self):
        (self.root / 'run').mkdir()
        with self.assertRaises(FileExistsError):
            bridge.run_task(self.task(), self.root, self.root / 'run')

    def test_invalid_budget(self):
        for value in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError):
                bridge.validate_task(self.task(timeout_seconds=value))

    def test_shell_string_not_accepted(self):
        task = self.task()
        task['steps'][0]['argv'] = 'echo ok'
        with self.assertRaises(ValueError):
            bridge.validate_task(task)

    def test_missing_executable_is_recorded(self):
        task = self.task()
        task['steps'][0]['argv'] = ['not-a-real-program-for-fixture-32842']
        result = bridge.run_task(task, self.root, self.root / 'run')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['steps'][0]['returncode'], 127)

    def test_packet_does_not_claim_quality_or_copy_log_secrets(self):
        bridge.run_task(self.task('print("PRIVATE_TOKEN_EXAMPLE")'), self.root, self.root / 'run')
        bridge.make_packet(self.root / 'run', self.root / 'packet')
        text = (self.root / 'packet' / 'reply.md').read_text()
        self.assertNotIn('PRIVATE_TOKEN_EXAMPLE', text)
        self.assertIn('未验证', text)

class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(preflight, 'preflight implementation is not present yet')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def png_header(self, path, bits=16):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I',13) + b'IHDR' +
                         struct.pack('>IIBBBBB',16,12,bits,2,0,0,0) + b'\0\0\0\0')

    def sample(self, split, stem='sample'):
        for role in ('raw_images','denoised_raw_images'):
            self.png_header(self.root / split / role / f'{stem}.png')
        p = self.root / split / 'srgb_images_style_0' / f'{stem}.jpg'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'not-decoded-in-header-only-audit')
        p = self.root / split / 'data' / f'{stem}.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'cam_illum':[1,1,1], 'ccm':[[1,0,0],[0,1,0],[0,0,1]]}))

    def test_missing_root_blocked_not_synthetic(self):
        result = preflight.audit_s24(None)
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['training_ready'])

    def test_pairs_train_val_only_by_default(self):
        self.sample('train','a'); self.sample('val','b')
        result = preflight.audit_s24(self.root)
        self.assertEqual(result['status'], 'header_checks_passed')
        self.assertEqual(result['splits']['train']['complete_pairs'], 1)
        self.assertNotIn('test', result['splits'])
        self.assertFalse(result['training_ready'])

    def test_missing_pair_is_reported(self):
        self.sample('train','a'); self.sample('val','b')
        (self.root / 'train/data/a.json').unlink()
        self.assertEqual(preflight.audit_s24(self.root)['status'], 'blocked')

    def test_png8_not_accepted_as_raw16(self):
        self.sample('train','a'); self.sample('val','b')
        self.png_header(self.root / 'train/raw_images/a.png', bits=8)
        self.assertEqual(preflight.audit_s24(self.root)['status'], 'blocked')

    def test_invalid_metadata_fails(self):
        self.sample('train','a'); self.sample('val','b')
        (self.root / 'train/data/a.json').write_text('{"cam_illum":[0,1,1],"ccm":[1,2,3]}')
        self.assertEqual(preflight.audit_s24(self.root)['status'], 'blocked')

    def test_stem_collision_is_blocked(self):
        self.sample('train','same'); self.sample('val','same')
        result = preflight.audit_s24(self.root)
        self.assertEqual(result['status'], 'blocked')
        self.assertTrue(result['cross_split_stem_collisions'])

    def test_source_audit_does_not_claim_cuda_execution(self):
        p = self.root / 'capture_tm'; p.mkdir()
        (p / 'joint_cli.py').write_text("parser.add_argument('--epochs')")
        (p / 'joint_experiment.py').write_text('def run_factorial(manifest, output):\n    pass\n')
        result = preflight.audit_source(self.root)
        self.assertFalse(result['device_cli_declared'])
        self.assertFalse(result['cuda_execution_verified'])

if __name__ == '__main__':
    unittest.main()
