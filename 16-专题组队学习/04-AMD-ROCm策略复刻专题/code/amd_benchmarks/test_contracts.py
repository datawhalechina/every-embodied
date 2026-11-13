"""Regression checks for run attribution and failed-stage reporting."""
import json
from pathlib import Path
import tempfile
import unittest
from .common import bind_run, fingerprint, reconcile_metrics, require_free_port, write_json
from .notebook_api import show_videos


class ContractTests(unittest.TestCase):
    def test_task_adapters_survive_model_selection_and_load_is_strict(self):
        import dataclasses
        from .dexjoco import checkpoint_model_config
        @dataclasses.dataclass
        class Model:
            paligemma_variant: str = 'gemma_2b_lora'
            action_expert_variant: str = 'gemma_300m_lora'
            def load(self, params, *, remove_extra_params=True):
                return remove_extra_params
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'params/_METADATA'
            path.parent.mkdir()
            write_json(path, {'tree_metadata': {'params/lora_a': {}, 'params/lora_b': {}}})
            model, audit = checkpoint_model_config(Model(), temp)
            self.assertEqual(audit['adapter_tensor_count'], 2)
            self.assertEqual(model.paligemma_variant, 'gemma_2b_lora')
            self.assertFalse(model.load({}))
            with self.assertRaisesRegex(ValueError, 'disables'):
                checkpoint_model_config(Model('gemma_2b', 'gemma_300m'), temp)
            write_json(path, {'tree_metadata': {'params/kernel': {}}})
            plain, _ = checkpoint_model_config(Model(), temp)
            self.assertEqual(plain.paligemma_variant, 'gemma_2b')

    def test_cold_evaluation_cannot_satisfy_prewarmed_recipe(self):
        from .pipeline import completed
        with tempfile.TemporaryDirectory() as temp:
            config = dict(run_root=temp, eval_seeds=list(range(100, 120)), warmup_policy=True)
            result = dict(seeds=config['eval_seeds'], validation_only=False)
            write_json(Path(temp) / 'baseline_status.json', dict(state='complete', result=result))
            self.assertFalse(completed(config, 'baseline'))
            result['warmup_policy'] = True
            write_json(Path(temp) / 'baseline_status.json', dict(state='complete', result=result))
            self.assertTrue(completed(config, 'baseline'))

    def test_changed_training_identity_cannot_reuse_results(self):
        with tempfile.TemporaryDirectory() as temp:
            c = dict(run_root=temp, benchmark='robocasa', task='A', batch_size=1,
                     learning_rate=1e-5, seed=42, source_root='/src', dataset_root='/data', base_model='/model')
            bind_run(c, {'sha256': 'dataset-a'}, {'sha256': 'model-a'})
            with self.assertRaises(RuntimeError):
                bind_run(dict(c, task='B'), {'sha256': 'dataset-a'}, {'sha256': 'model-a'})
            with self.assertRaises(RuntimeError):
                bind_run(c, {'sha256': 'dataset-b'}, {'sha256': 'model-a'})

    def test_weight_contents_determine_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'weights.bin'
            p.write_bytes(b'first')
            first = fingerprint(temp, ['*.bin'])
            p.write_bytes(b'other')
            second = fingerprint(temp, ['*.bin'])
            self.assertNotEqual(first['sha256'], second['sha256'])
            self.assertEqual(first['files'][0]['bytes'], second['files'][0]['bytes'])

    def test_empty_model_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                fingerprint(temp, ['*.safetensors'])

    def test_old_evaluation_cannot_show_as_new_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            write_json(Path(temp) / 'trained_model.json', {'checkpoint_sha256': 'new'})
            write_json(Path(temp) / 'evaluate_status.json', {
                'state': 'complete', 'result': {'checkpoint_sha256': 'old'}})
            with self.assertRaisesRegex(RuntimeError, 'Weights changed'):
                show_videos({'run_root': temp})

    def test_occupied_server_port_is_rejected(self):
        import socket
        with socket.socket() as endpoint:
            endpoint.bind(('127.0.0.1', 0))
            endpoint.listen()
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                require_free_port(endpoint.getsockname()[1])

    def test_baseline_has_no_fabricated_training_record(self):
        from . import dexjoco, robocasa
        with tempfile.TemporaryDirectory() as temp:
            write_json(Path(temp) / 'identity.json', {'base_model_identity': {'sha256': 'base-identity'}})
            config = {'run_root': temp, 'base_model': '/frozen/base', 'policy_variant': 'base'}
            for module in [dexjoco, robocasa]:
                record = module.policy_record(config)
                self.assertEqual(record['policy_variant'], 'base')
                self.assertEqual(record['checkpoint_sha256'], 'base-identity')
            self.assertFalse((Path(temp) / 'trained_model.json').exists())

    def test_resume_archives_uncommitted_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'metrics.jsonl'
            path.write_text('{"step":1,"loss":1}\n{"step":2,"loss":0.5}\n{"step":3')
            result = reconcile_metrics(temp, 1)
            self.assertEqual([json.loads(x)['step'] for x in path.read_text().splitlines()], [1])
            self.assertTrue(Path(result['raw_attempt_log']).is_file())


if __name__ == '__main__':
    unittest.main()
