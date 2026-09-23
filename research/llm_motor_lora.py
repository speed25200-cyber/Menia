"""mlx-lm LoRA in raw completion, with the loss on the completion only (docs/LLM_MOTOR_OBJECTIVE_PROTOCOL.md).

mlx-lm reads {"prompt", "completion"} examples through its chat template; the tests of the adjusted body read
the model in raw completion. This wrapper encodes each example as raw text, prompt then completion, and masks
the prompt in the loss, then runs mlx_lm.lora with the given arguments.
"""
import sys


def raw_process(self, d):
    prompt = self.tokenizer.encode(d[self.prompt_key], add_special_tokens=False)
    completion = self.tokenizer.encode(d[self.completion_key], add_special_tokens=False)
    return (list(prompt) + list(completion), len(prompt))


def main(argv=None):
    from mlx_lm.tuner import datasets
    from mlx_lm import lora
    datasets.CompletionsDataset.process = raw_process
    sys.argv = ["mlx_lm.lora"] + list(sys.argv[1:] if argv is None else argv)
    lora.main()


if __name__ == "__main__":
    main()
