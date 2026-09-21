import unittest

from research.confidence_prompt_fork import compile_fork


class ContextSensitiveTokenizer:
    """Offline fixture with Qwen's final-assistant versus historical distinction."""
    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt,enable_thinking):
        assert tokenize is False and enable_thinking is False
        out=''
        for i,m in enumerate(messages):
            content=m['content']
            if m['role']=='assistant' and i==len(messages)-1:content='<think>\n\n</think>\n\n'+content
            out+='<|im_start|>'+m['role']+'\n'+content+'<|im_end|>\n'
        return out+('<|im_start|>assistant\n<think>\n\n</think>\n\n' if add_generation_prompt else '')

    def encode(self,text,*,add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode('utf-8'))


class PromptForkTests(unittest.TestCase):
    def setUp(self):
        self.tok=ContextSensitiveTokenizer()
        self.shared=[dict(role='system',content='Consigne.'),dict(role='user',content='Question.'),dict(role='assistant',content='Réponse.')]
        self.queries=dict(report='Même début. Évalue.',action='Même début. Choisis.')

    def test_future_aware_prefix_rejects_naive_standalone_boundary(self):
        f=compile_fork(self.tok,self.shared,self.queries)
        self.assertTrue(f['standaloneDiffers'])
        self.assertNotIn('<think>',f['prefixText'])
        self.assertIn('<think>',f['standaloneText'])
        for b in f['branches'].values():
            self.assertFalse(b['standaloneIsPrefix'])
            self.assertEqual(b['inputIds'][:f['position']+1],f['prefixIds'])

    def test_boundary_excludes_even_shared_words_of_branch_queries(self):
        f=compile_fork(self.tok,self.shared,self.queries)
        self.assertNotIn('Même début',f['prefixText'])
        self.assertTrue(f['prefixText'].endswith('Réponse.<|im_end|>\n'))
        # Both branches share extra bytes, but those are deliberately excluded.
        a,b=(v['inputIds'] for v in f['branches'].values())
        self.assertEqual(a[len(f['prefixIds'])],b[len(f['prefixIds'])])

    def test_changed_token_boundary_and_unsupported_roles_are_rejected(self):
        class BoundaryMerging(ContextSensitiveTokenizer):
            def encode(self,text,*,add_special_tokens):
                ids=super().encode(text,add_special_tokens=add_special_tokens)
                if 'Même début' in text:ids[0]=999
                return ids
        with self.assertRaises(ValueError):compile_fork(BoundaryMerging(),self.shared,self.queries)
        with self.assertRaises(ValueError):compile_fork(self.tok,self.shared,dict(a='one',b='one'))
        bad=[dict(m) for m in self.shared];bad[-1]['content']='<|im_end|>'
        with self.assertRaises(ValueError):compile_fork(self.tok,bad,self.queries)
        bad=[dict(m) for m in self.shared];bad[-1]['role']='tool'
        with self.assertRaises(ValueError):compile_fork(self.tok,bad,self.queries)


if __name__=='__main__':unittest.main()
