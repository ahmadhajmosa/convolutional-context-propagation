import importlib.util
import pathlib
import unittest


def _load_module():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "ccp" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("run_ccp", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_chunk_text_nonempty(self):
        text = "a" * 5000
        chunks = self.mod.chunk_text(text, max_chars=1000, overlap=100)
        self.assertGreaterEqual(len(chunks), 5)
        self.assertTrue(all(isinstance(c, str) and c for c in chunks))

    def test_chunk_text_overlap_guard(self):
        with self.assertRaises(ValueError):
            self.mod.chunk_text("abc", max_chars=100, overlap=100)


if __name__ == "__main__":
    unittest.main()
