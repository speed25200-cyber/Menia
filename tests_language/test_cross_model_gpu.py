"""Offline API checks with a tiny RANDOM Qwen3, never scores of the real models."""
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from research.cross_model_gpu import generate_text
from research.cross_model_prediction import SETTINGS


class CrossModelBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        tok = Tokenizer(WordLevel({"[UNK]":0, "[PAD]":1, "[EOS]":2, "OFF":3, "ON":4, "hello":5}, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        cls.tokenizer.chat_template = "{{ 'ON' if enable_thinking else 'OFF' }} {% for m in messages %}{{ m['content'] }} {% endfor %}"
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             max_position_embeddings=256, eos_token_id=2, pad_token_id=1)
        with tempfile.TemporaryDirectory() as tmp:
            Qwen3ForCausalLM(config).save_pretrained(tmp, safe_serialization=True)
            cls.model = AutoModelForCausalLM.from_pretrained(tmp, use_safetensors=True, attn_implementation="sdpa").eval()

    def test_actual_qwen_generation_api_and_repeatable_sampling(self):
        settings = dict(SETTINGS, max_new_tokens=6)
        messages = [dict(role="user", content="hello")]
        text, metrics = generate_text(self.model, self.tokenizer, messages, 12, settings=settings)
        repeat, again = generate_text(self.model, self.tokenizer, messages, 12, settings=settings)
        self.assertEqual(text, repeat)
        self.assertEqual(metrics, again)
        self.assertEqual(metrics["inputTokens"], 2)
        self.assertLessEqual(metrics["outputTokens"], 6)
        self.assertTrue(metrics["effectiveGeneration"]["do_sample"])
        self.assertEqual(metrics["effectiveGeneration"]["temperature"], .7)

    def test_input_overflow_rejected_without_truncation(self):
        with self.assertRaises(ValueError):
            generate_text(self.model, self.tokenizer, [dict(role="user", content="hello hello")], 12,
                          settings=dict(SETTINGS, max_input_tokens=2))

    def test_disabled_thinking_is_passed_to_the_template(self):
        text = self.tokenizer.apply_chat_template([dict(role="user", content="hello")], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=SETTINGS["enable_thinking"])
        self.assertTrue(text.startswith("OFF"))


if __name__ == "__main__":
    unittest.main()
