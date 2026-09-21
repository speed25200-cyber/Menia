"""Single pinned Qwen3-4B; each native choice and answer is a fresh invocation."""
import os

from research.cross_model_gpu import environment, generate_text
from research.cross_model_prediction import MODELS, SETTINGS
from research.native_choice_plan import DECISION_SETTINGS
from research.native_choice_journal import source_hash


class NativeChoiceBackend:
    origin = 'transformers_gpu'

    def __init__(self):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        os.environ['TOKENIZERS_PARALLELISM']='false'
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.metadata=environment()
        self.metadata.update(models={'A':MODELS['A']},settings=dict(choice=DECISION_SETTINGS,answer=SETTINGS),
                             nativeChoiceSourceHash=source_hash(),newWeightUpdates=0)
        spec=MODELS['A']
        self.tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],torch_dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval()
        if self.model.config._commit_hash!=spec['revision']:
            raise RuntimeError('Model revision differs')

    def generate(self, request):
        return generate_text(self.model,self.tokenizer,request['messages'],request['call']['seed'],settings=request['settings'])


if __name__=='__main__':
    import argparse
    import json
    from pathlib import Path
    from research.native_choice_journal import collect, verify_calibration_parent
    from research.native_choice_analysis import analyze
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path)
    parser.add_argument('--parent',type=Path,required=True)
    args=parser.parse_args()
    parent=verify_calibration_parent(args.parent)
    backend=NativeChoiceBackend()
    backend.metadata['verifiedCalibrationParentSHA256']=parent
    collect(args.journal,backend)
    report=analyze(args.journal)
    args.journal.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('complete','recorded','planned','statuses','actualLLMCalls','actualToolCalls')}),flush=True)
