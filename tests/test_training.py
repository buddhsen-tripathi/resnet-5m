import contextlib
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model import MAX_PARAMS, ResNet, count_params
import train


class TinyCifar(Dataset):
    def __init__(self, *args, transform, **kwargs):
        self.transform = transform

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return self.transform(Image.new('RGB', (32, 32), (80 + index, 130, 200))), index % 2


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_supcon_matches_pairwise_definition_and_has_finite_gradients(self):
        features = torch.randn(4, 2, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 2])
        temperature = 0.2
        z = F.normalize(features, dim=-1).reshape(8, 8)
        classes = labels.repeat_interleave(2)
        terms = []
        for anchor in range(8):
            scores = torch.stack([z[anchor].dot(z[j]) / temperature for j in range(8) if j != anchor])
            normalizer = torch.logsumexp(scores, dim=0)
            positive = [normalizer - z[anchor].dot(z[j]) / temperature
                        for j in range(8) if j != anchor and classes[j] == classes[anchor]]
            terms.append(torch.stack(positive).mean())
        expected = torch.stack(terms).mean()
        actual = train.supervised_contrastive_loss(features, labels, temperature)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.norm().item(), 0)

    def test_supcon_excludes_self_and_rejects_mixed_labels(self):
        features = torch.ones(3, 2, 4)
        loss = train.supervised_contrastive_loss(features, torch.zeros(3, dtype=torch.long))
        self.assertAlmostEqual(loss.item(), math.log(5), places=5)
        with self.assertRaises(ValueError):
            train.supervised_contrastive_loss(features, torch.ones(3, 10) / 10)

    def test_two_augmix_views_keep_eight_pixel_cutout(self):
        with patch.object(sys, 'argv', ['train.py', '--augmix', '--supcon', '0.1', '--cutout',
                                      '--cutout-size', '8', '--workers', '0', '--batch-size', '4']):
            args = train.get_args()
        with patch.object(train.datasets, 'CIFAR10', TinyCifar):
            loader, _ = train.loaders(args)
        views, labels = next(iter(loader))
        self.assertEqual(labels.shape, (4,))
        self.assertEqual(len(views), 2)
        for view in views:
            self.assertEqual(view.shape, (4, 3, 32, 32))
            self.assertTrue(torch.isfinite(view).all())
            self.assertTrue(((view >= 0) & (view <= 1)).all())
            torch.testing.assert_close((view[:, 0] == 0.5).sum((1, 2)), torch.full((4,), 64))

    def test_resample_keeps_size_and_changes_pixels(self):
        image = Image.fromarray(torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8).numpy())
        self.assertIs(train.RandomResample(0.0)(image), image)
        torch.manual_seed(0)
        outputs = [train.RandomResample(1.0, (48, 64))(image) for _ in range(8)]
        for output in outputs:
            self.assertEqual(output.size, (32, 32))
            self.assertEqual(output.mode, 'RGB')
        self.assertTrue(any(list(output.getdata()) != list(image.getdata()) for output in outputs))

    def test_shift_augmentation_flags_build_valid_views(self):
        with patch.object(sys, 'argv', ['train.py', '--augmix', '--supcon', '0.1', '--cutout', '--cutout-size', '8',
                                      '--resample', '1', '--jitter', '0.2', '--jitter-p', '1',
                                      '--workers', '0', '--batch-size', '4']):
            args = train.get_args()
        with patch.object(train.datasets, 'CIFAR10', TinyCifar):
            loader, _ = train.loaders(args)
        views, _ = next(iter(loader))
        for view in views:
            self.assertEqual(view.shape, (4, 3, 32, 32))
            self.assertTrue(((view >= 0) & (view <= 1)).all())
        for bad in (['--resample', '1.5'], ['--resample-size', '64', '48'], ['--jitter', '-0.1']):
            with patch.object(sys, 'argv', ['train.py', *bad]), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                train.get_args()

    def test_projection_fits_budget_and_inference_loads_without_it(self):
        model = ResNet(widths=[128, 192, 228], blocks=[4, 3, 2], se=False).eval()
        head = torch.nn.Sequential(torch.nn.Linear(228, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128))
        self.assertEqual(count_params(model) + count_params(head), 4_983_618)
        self.assertLessEqual(count_params(model) + count_params(head), MAX_PARAMS)
        with torch.no_grad():
            x = torch.rand(2, 3, 32, 32)
            logits, features = model(x, return_features=True)
            torch.testing.assert_close(logits, model(x), rtol=0, atol=0)
            torch.testing.assert_close(logits, model.fc(features), rtol=0, atol=0)
        clone = ResNet(widths=[128, 192, 228], blocks=[4, 3, 2], se=False)
        clone.load_state_dict(model.state_dict(), strict=True)

    def test_budget_check_blocks_students_and_admits_flagged_teachers(self):
        wide = count_params(ResNet(widths=[160, 320, 640], blocks=[4, 4, 4], se=False))
        self.assertEqual(wide, 36_688_170)
        train.check_budget(MAX_PARAMS)
        train.check_budget(4_983_618, allow_over_budget=True)
        with self.assertRaisesRegex(AssertionError, "allow-over-budget"):
            train.check_budget(wide)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            train.check_budget(wide, allow_over_budget=True)
        self.assertIn("must not be submitted", out.getvalue())
        with patch.object(sys, "argv", ["train.py", "--allow-over-budget"]):
            self.assertTrue(train.get_args().allow_over_budget)
        with patch.object(sys, "argv", ["train.py"]):
            self.assertFalse(train.get_args().allow_over_budget)

    def test_combined_training_resumes_with_projection_optimizer_and_rng(self):
        generator = torch.Generator().manual_seed(9)
        x = torch.rand(8, 3, 32, 32, generator=generator)
        y = torch.tensor([0, 1, 0, 1, 2, 2, 3, 3])
        paired = TensorDataset(x, x.flip(3), y)

        class ViewsDataset(Dataset):
            def __len__(self):
                return len(paired)

            def __getitem__(self, index):
                first, second, label = paired[index]
                return (first, second), label

        def small_loaders(args):
            return DataLoader(ViewsDataset(), batch_size=4), DataLoader(TensorDataset(x, y), batch_size=4)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cfg = dict(widths=[4, 8], blocks=[1, 1], se=False, block='basic', drop_path=0.0)
            teacher = directory / 'teacher.pt'
            torch.save({'model': ResNet(**cfg).state_dict(), 'cfg': cfg}, teacher)
            full, resumed = directory / 'full.pt', directory / 'resumed.pt'
            base = ['train.py', '--widths', '4', '8', '--blocks', '1', '1', '--no-se',
                    '--epochs', '2', '--warmup-epochs', '1', '--batch-size', '4', '--workers', '0',
                    '--augmix', '--cutout', '--cutout-size', '8', '--supcon', '0.1', '--mix', '1',
                    '--kd-alpha', '0.7', '--label-smoothing', '0', '--teacher', str(teacher)]
            with patch.object(train, 'loaders', small_loaders), \
                 patch.object(torch.cuda, 'is_available', return_value=False), \
                 patch.object(torch.backends.mps, 'is_available', return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch.object(train, 'evaluate', return_value=0.5), \
                     patch.object(sys, 'argv', base + ['--out', str(full)]):
                    train.main()
                with patch.object(train, 'evaluate', side_effect=[0.5, InterruptedError('simulated stop')]), \
                     patch.object(sys, 'argv', base + ['--out', str(resumed)]):
                    with self.assertRaises(InterruptedError):
                        train.main()
                saved = torch.load(resumed, weights_only=True)
                self.assertEqual(saved['epoch'], 1)
                self.assertIn('projector', saved)
                self.assertIn('rng', saved)
                with patch.object(train, 'evaluate', return_value=0.5), \
                     patch.object(sys, 'argv', base + ['--out', str(resumed), '--resume']):
                    train.main()
            expected = torch.load(full, weights_only=True)
            actual = torch.load(resumed, weights_only=True)
            self.assertEqual(actual['epoch'], 2)
            self.assertEqual(actual['sched'], expected['sched'])
            for group in ['model', 'projector']:
                for key in expected[group]:
                    torch.testing.assert_close(actual[group][key], expected[group][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
