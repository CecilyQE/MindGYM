"""Fresh generative distribution validation script smoke tests."""

import tempfile
import unittest
from pathlib import Path

from scripts.validate_generative_distribution import main


class GenerativeDistributionValidationTests(unittest.TestCase):
    def test_smoke_subset_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            json_path = out_dir / "validation.json"
            md_path = out_dir / "validation.md"
            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "validate_generative_distribution.py",
                    "--experiments",
                    "badham2017deficits/exp1.csv",
                    "peterson2021using/exp1.csv",
                    "wilson2014humans/exp1.csv",
                    "--seeds",
                    "2",
                    "--max-steps",
                    "80",
                    "--json",
                    str(json_path),
                    "--md",
                    str(md_path),
                ]
                main()
            finally:
                sys.argv = old_argv

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn("Generative Distribution Validation", md_path.read_text())


if __name__ == "__main__":
    unittest.main()
