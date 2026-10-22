"""Static checks only; these do not validate training or numerical behavior."""
import ast
import importlib.util
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceChecks(unittest.TestCase):
    def test_python_syntax(self):
        for path in ROOT.rglob('*.py'):
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(), filename=str(path))

    def test_paper_entry_points(self):
        expected = {
            'models/segmentor/semalign.py': {'SemAlign'},
            'models/decode_heads/semalign_head.py': {
                'SemAlignHead', 'LAA', 'LanguageGuidedAnisotropicAttention',
                'ContextAwareTextAdaptation', 'ContextAwareTextAdaptationLite'},
            'models/losses/rcs_loss.py': {'rcs_loss', 'SegLossPlusRCS'},
        }
        for filename, names in expected.items():
            tree = ast.parse((ROOT / filename).read_text())
            actual = {node.name for node in tree.body
                      if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
            self.assertTrue(names <= actual, filename)

    def test_saved_configuration_values_preserved(self):
        for dataset, classes, cata, iters in [('voc', 15, False, 20000),
                                              ('coco', 156, True, 80000)]:
            cfg = runpy.run_path(str(ROOT / 'configs' /
                                    ('semalign_' + dataset + '_snapshot.py')))
            self.assertNotIn('_base_', cfg)
            self.assertEqual(cfg['model']['type'], 'SemAlign')
            head = cfg['model']['decode_head']
            self.assertEqual(head['type'], 'SemAlignHead')
            self.assertEqual(head['use_cata'], cata)
            self.assertEqual(head['laa_stages'], [2, 3])
            self.assertTrue(head['use_ccp'])
            self.assertEqual(head['loss_decode']['num_classes'], classes)
            self.assertEqual(head['loss_decode']['type'], 'SegLossPlusRCS')
            self.assertEqual(head['loss_decode']['mask_weight'], 100.0)
            self.assertEqual(head['loss_decode']['dice_weight'], 1.0)
            self.assertEqual(cfg['runner']['max_iters'], iters)

    def test_preflight_identifies_missing_config(self):
        spec = importlib.util.spec_from_file_location('preflight', ROOT / 'tools/preflight.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = module.check(['configs/does_not_exist.py'], check_dependencies=False)
        self.assertIn('Configuration: configs/does_not_exist.py', missing)

    def test_no_successful_exit_on_laa_failure(self):
        tree = ast.parse((ROOT / 'models/decode_heads/semalign_head.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, 'exit')

    def test_preflight_checks_list_valued_dataset_paths(self):
        spec = importlib.util.spec_from_file_location('preflight', ROOT / 'tools/preflight.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = module.check(['configs/semalign_voc_snapshot.py'], check_dependencies=False)
        expected = ROOT / 'datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/aug.txt'
        if not expected.exists():
            self.assertIn('Required path: datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/aug.txt', missing)


if __name__ == '__main__':
    unittest.main()
