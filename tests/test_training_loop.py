"""CPU regressions for training limits and optional diagnostics."""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from pyCAESAR import train_vae3d as training


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs):
        return {"output": inputs * self.weight, "bpp": self.weight.square()}


class TrainingLoopTests(unittest.TestCase):
    def test_debug_flags(self):
        for flags, expected in (([], False), (["--debug"], True), (["--no-debug"], False)):
            with patch.object(sys, "argv", ["train", *flags]):
                self.assertEqual(training.get_argument().debug, expected)

    def test_iteration_limit_and_debug(self):
        batch = {"input": torch.ones(1, 2, 2, 2), "offset": torch.tensor(0.),
                 "scale": torch.tensor(1.), "index": torch.tensor(0)}
        loader = DataLoader([batch] * 5, batch_size=2)
        for debug in (False, True):
            model = ToyModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
            with patch.object(training, "check_finite", wraps=training.check_finite) as check:
                mse, bpp, iteration = training.train_epoch_vae(
                    model, loader, optimizer, scheduler, torch.nn.MSELoss(), 0.1,
                    torch.device("cpu"), iteration=99, debug=debug, max_iterations=100,
                )
            self.assertEqual(iteration, 100)
            self.assertEqual(scheduler.last_epoch, 1)
            self.assertAlmostEqual(mse, 0.25)
            self.assertAlmostEqual(bpp, 0.025)
            self.assertEqual(check.called, debug)

    def test_completion_progress(self):
        # Exercise the actual main-loop expression at and beyond completion.
        tree = ast.parse(Path(training.__file__).read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "eval_index"
                                  for t in node.targets))
        expression = compile(ast.Expression(assignment.value), "progress", "eval")
        for current, expected in ((99, 99), (100, 100), (150, 100)):
            self.assertEqual(eval(expression, {"cur_iters": current,
                             "args": type("Args", (), {"iterations": 100})()}), expected)


if __name__ == "__main__":
    unittest.main()
