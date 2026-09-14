"""Optional local Qwen verbalization, sharing the recorded agent context."""
import time


def quantize_cpu(module):
    """Convert one linear layer at a time, avoiding a full FP32 model copy."""
    import torch
    from torch.ao.nn.quantized.dynamic import Linear
    from torch.ao.quantization import default_dynamic_qconfig
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            child.float()
            child.qconfig = default_dynamic_qconfig
            setattr(module, name, Linear.from_float(child))
        else:
            quantize_cpu(child)
    # Remaining local parameters belong to embeddings or normalization layers.
    # Recurse=False avoids changing the dtype of packed quantized children.
    for parameter in module.parameters(recurse=False):
        parameter.data = parameter.data.float()
    for name, buffer in module.named_buffers(recurse=False):
        if buffer.is_floating_point():
            module._buffers[name] = buffer.float()


class LocalGenerator:
    def __init__(self, model='Qwen/Qwen3-1.7B', revision='main', *, threads=4, cpu_precision='float32'):
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if type(threads) is not int or threads < 1:
            raise ValueError('Positive thread count required')
        if cpu_precision not in ('float32', 'int8'):
            raise ValueError('CPU precision must be float32 or int8')
        torch.set_num_threads(threads)
        self.model_id = model
        self.revision = HfApi().model_info(model, revision=revision).sha
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=self.revision)
        cpu = not torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=self.revision,
            torch_dtype=torch.bfloat16,
            device_map={'': 'cpu'} if cpu else 'auto', low_cpu_mem_usage=True)
        self.model.eval()
        self.quantization = 'none'
        if cpu and cpu_precision == 'int8':
            # Avoid emulated BF16 on CPUs without native support and keep the
            # resident model smaller. The original cached weights stay intact.
            quantize_cpu(self.model)
            self.quantization = 'dynamic-int8-linear'
        elif cpu:
            # Load native BF16 weights first, then convert without the loader's
            # simultaneous dtype conversion and meta-parameter assignment.
            self.model.float()
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
            'dtype': str(self.model.dtype), 'quantization': self.quantization,
            'device': str(self.model.device)}
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
