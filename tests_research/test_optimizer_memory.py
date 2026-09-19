import itertools
import unittest

from research import optimizer_memory as study


class OptimizerMemoryDesignTests(unittest.TestCase):
    def test_forks_share_exact_old_schedule_and_evaluation_is_paired(self):
        fixed=study.plan();old=study.previous.plan()
        self.assertEqual(len(fixed['evaluation']),29184)
        self.assertEqual(len(fixed['training']),12)
        self.assertEqual(sum(len(u['groups']) for u in fixed['training']),288)
        for rep in range(3):
            groups=next(u['groups'] for u in old['training'] if u['key']==f'r{rep}-composed')
            arms={u['arm']:u for u in fixed['training'] if u['replication']==rep}
            self.assertEqual(arms['prefix']['groups'],groups[:48])
            for arm in study.ARMS[1:]:self.assertEqual(arms[arm]['groups'],groups[48:])
            self.assertTrue(all(g['monitorFamily']=='visible' for g in groups[48:]))
        conditions={arm:{tuple((k,v) for k,v in r.items() if k not in ('arm','id'))
                         for r in fixed['evaluation'] if r['arm']==arm} for arm in study.ARMS}
        self.assertTrue(all(v==conditions['prefix'] for v in conditions.values()))

    def test_fresh_sentences_and_noises_and_reserved_content_words(self):
        old=study.diagnostic.old_blocks()+study.diagnostic.plan()['blocks']
        used={s for b in old for s in b['sentences']};noises={b['noiseSeed'] for b in old}
        fresh=[b for b in study.plan()['blocks'] if b['split']!='train']
        sentences=[s for b in fresh for s in b['sentences']]
        self.assertFalse(used.intersection(sentences));self.assertEqual(len(sentences),len(set(sentences)))
        self.assertFalse(noises.intersection(b['noiseSeed'] for b in fresh))
        self.assertEqual(len(fresh),len({b['noiseSeed'] for b in fresh}))
        # Content lexicon is reserved relative to the fine-tuning bank, never claimed unseen in pretraining.
        old_content={x.split()[-1] for bank in study.SAME_LEXICON for x in bank}
        new_content={x.split()[-1] for bank in study.LEXICON for x in bank}
        self.assertFalse(old_content & new_content)
        for rep,split in itertools.product(range(3),('test','lexical')):
            bs=[b for b in fresh if b['replication']==rep and b['split']==split]
            self.assertEqual(sum(b['marker']==0 for b in bs),len(bs)//2)


if __name__=='__main__':unittest.main()
