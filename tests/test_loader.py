"""Tests for NER data loading."""
import unittest
from src.data_loader import CoNLLLoader

class TestCoNLLLoader(unittest.TestCase):
    def test_label_encoding(self):
        loader = CoNLLLoader(["O", "B-PER", "I-PER"])
        self.assertEqual(loader.label2id["O"], 0)
        self.assertEqual(loader.label2id["B-PER"], 1)

    def test_build_vocab(self):
        loader = CoNLLLoader()
        vocab = loader.build_vocab([["Hello", "world"], ["test"]])
        self.assertIn("hello", vocab)
        self.assertIn("<PAD>", vocab)

if __name__ == "__main__":
    unittest.main()
