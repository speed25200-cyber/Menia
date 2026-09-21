"""Actual random Qwen computation with 36 tiny layers, never a Menia score."""
import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from research.state_interchange_ops import capture_forward,patch_forward,tensor_hash


class InterchangeOperationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(61)
        self.model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=8,intermediate_size=16,
            num_hidden_layers=36,num_attention_heads=2,num_key_value_heads=1,head_dim=4)).eval()
        self.inputs=dict(input_ids=torch.tensor([[1,2,3,4,5,6]]),attention_mask=torch.ones(1,6,dtype=torch.long))
        self.spans=[[1,2],[3,4]];self.block={'noiseSeed':27}

    def test_shams_are_exact_and_last_layer_copies_donor_logits(self):
        with torch.inference_mode():
            a,_,ha=capture_forward(self.model,self.inputs,self.spans,self.block,0,(17,23,35))
            b,_,hb=capture_forward(self.model,self.inputs,self.spans,self.block,1,(17,23,35))
            self.assertTrue(torch.equal(ha[17],hb[17]))
            self.assertFalse(torch.equal(ha[23],hb[23]))
            for site in (17,23,35):
                actual,_,stats=patch_forward(self.model,self.inputs,self.spans,self.block,0,site,ha[site])
                self.assertTrue(torch.equal(a,actual));self.assertEqual(stats['displacementNorm'],0.)
            actual,_,stats=patch_forward(self.model,self.inputs,self.spans,self.block,0,35,hb[35])
            self.assertTrue(torch.equal(b,actual))
            self.assertEqual(stats['donorHash'],tensor_hash(hb[35]))
        self.assertTrue(all(not m._forward_hooks for m in self.model.modules()))

    def test_wrong_donor_fails_and_removes_all_hooks(self):
        with torch.inference_mode(),self.assertRaisesRegex(ValueError,'shape, dtype or device'):
            patch_forward(self.model,self.inputs,self.spans,self.block,1,23,torch.zeros(7))
        self.assertTrue(all(not m._forward_hooks for m in self.model.modules()))

    def test_complete_actual_qwen_group_and_distinct_prefix_lengths(self):
        from tests_language.test_cross_model_gpu import CrossModelBackendTests
        from research.state_interchange_gpu import run_group
        CrossModelBackendTests.setUpClass()
        tokenizer=CrossModelBackendTests.tokenizer
        pair={role:dict(id=role,noiseSeed=27+i,marker=i,positivePosition=i+1,
                       sentences=['hello hello','hello hello hello']) for i,role in enumerate(('donor','recipient'))}
        outputs=run_group(self.model,tokenizer,pair,[3,4,5,6])
        self.assertEqual(len(outputs),136)
        longer=dict(input_ids=torch.tensor([[2,3,4,5,6,7,8,9]]),attention_mask=torch.ones(1,8,dtype=torch.long))
        with torch.inference_mode():
            donor,_,states=capture_forward(self.model,self.inputs,self.spans,self.block,1,(35,))
            patched,_,_=patch_forward(self.model,longer,self.spans,self.block,0,35,states[35])
            self.assertTrue(torch.equal(donor,patched))
        self.assertTrue(all(not m._forward_hooks for m in self.model.modules()))


if __name__=='__main__':unittest.main()
