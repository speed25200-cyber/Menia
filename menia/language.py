"""Optional local Qwen verbalization, sharing the recorded agent context."""
import time


class LocalGenerator:
    def __init__(self, model='Qwen/Qwen3-1.7B', revision='main', *, threads=4):
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if type(threads) is not int or threads < 1:
            raise ValueError('Positive thread count required')
        torch.set_num_threads(threads)
        self.model_id = model
        self.revision = HfApi().model_info(model, revision=revision).sha
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=self.revision,
            torch_dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True)
        self.model.eval()
        self.last_metrics = None

    def __call__(self, messages):
        import torch
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False).to(self.model.device)
        length = inputs.input_ids.shape[-1]
        if length > 1856:
            raise ValueError(f'Contexte trop long ({length} tokens ; maximum 1856). Réduire la question ou le contexte.')
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=192, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        generated = outputs[0, length:]
        self.last_metrics = {'model': self.model_id, 'revision': self.revision,
            'input_tokens': length, 'output_tokens': len(generated),
            'seconds': time.perf_counter()-started,
            'reached_token_limit': len(generated) == 192,
            'dtype': str(self.model.dtype), 'device': str(self.model.device)}
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
