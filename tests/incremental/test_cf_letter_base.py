"""Base acceptance checks: pure data tests plus mandatory real CPU model tests."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',CUDA_VISIBLE_DEVICES='')
import numpy as np
from genrec.cf_letter_protocol import allocate,encode_rows,gpt_arrays,metrics,plan,read,save,sha,load_base,submission,encoded

HAVE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


class Protocol(unittest.TestCase):
    def test_submission_matches_installed_generic_runtime_without_host_changes(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'scripts/lab_buffer'))
        import task_runtime
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save(root/'task-runtime-policy.json',dict(protocol_version=1,max_parallel=2,read_roots=[str(root)]))
            runner=SimpleNamespace(root=root,config=dict(task_runtime_version=1,worker_id='server2'),
                read=lambda code,path:(ROOT/path).read_bytes() if (ROOT/path).is_file() else None)
            values=submission('a'*40,'b'*64,root,ROOT)
            task_runtime.validate(runner,values['request'],encoded(values['request']),values['contract'],values['scope'])
            self.assertEqual(values['scope']['max_parallel'],1)
            self.assertEqual(values['contract']['optimizer_step_reservation'],54640)
            invalid=copy.deepcopy(values['contract']);invalid['optimizer_step_reservation']+=1
            with self.assertRaises(ValueError):
                task_runtime.validate(runner,values['request'],encoded(values['request']),invalid,values['scope'])

    def test_base_scope_budget_and_exclusions(self):
        p=plan(ROOT)
        self.assertEqual(p['optimizer_step_cap'],54640)
        self.assertEqual(p['allowed_splits'],['base_train','base_val'])
        self.assertFalse(p['future_evaluation']);self.assertFalse(p['update_training'])

    def row(self,seq=None,times=None):
        return dict(sample_id='a',user='raw-user',item_seq=seq or ['old','missing','old'],
            time_seq=times or [1,2,3],target_time=3)

    def test_truncate_raw_events_before_masking_and_keep_item_sid_zero_distinct(self):
        values=encode_rows([self.row()],{'old':1},2,8)
        self.assertEqual(values['history'].tolist(),[[1,0]])
        mapping,_=allocate([[0,0,0]],4)
        ids,mask,target=gpt_arrays(values,mapping,4,8)
        self.assertEqual(ids[0,1],0);self.assertTrue(mask[0,1])
        self.assertTrue(np.all(~mask[0,-4:]));self.assertTrue(np.all(ids[0,-4:]==33))
        self.assertEqual(target.tolist(),mapping.tolist())

    def test_same_time_history_and_duplicate_samples_rejected(self):
        with self.assertRaisesRegex(ValueError,'Noncausal'):encode_rows([self.row(times=[1,3,3])],{'old':1},2,8)
        with self.assertRaises(ValueError):encode_rows([self.row(),self.row()],{'old':1},2,8)

    def test_suffix_determinism_capacity_and_bijection(self):
        rows,stats=allocate([[0,1,2]]*4,4)
        self.assertEqual(rows[:,-1].tolist(),[12,13,14,15])
        self.assertEqual(stats['maximum_collision_bucket'],4)
        with self.assertRaisesRegex(ValueError,'capacity exhausted'):allocate([[0,1,2]]*5,4)

    def test_full_denominator_and_ndcg_are_independently_checkable(self):
        rows=[dict(target=2,predictions=[1,2],nll=3.),dict(target=0,predictions=[],nll=None)]
        result=metrics(rows,[1,2])
        self.assertEqual(result['full_denominator']['recall@2'],.5)
        self.assertEqual(result['conditional_on_servable']['recall@2'],1.)
        self.assertAlmostEqual(result['full_denominator']['ndcg@2'],.5/np.log2(3))

    def test_bound_loader_never_opens_future_and_detects_changed_base_bytes(self):
        from genrec.temporal import build_temporal_protocol
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);project=root/'project';directory=project/'cache/incremental/Beauty/beauty_20261007T104452Z';directory.mkdir(parents=True)
            rows=[('u'+str(i%8),'item'+str(i%32),1000000000+i,None) for i in range(500)]
            protocol=build_temporal_protocol(rows)
            save(directory/'protocol_manifest.json',protocol['manifest']);save(directory/'registries.json',protocol['registries'])
            for split in ('base_train','base_val'):
                (directory/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in protocol['examples'][split]))
            (directory/'future_test.jsonl').write_text('DO NOT OPEN\n')
            hashes={f.name:sha(f) for f in directory.iterdir() if f.name!='future_test.jsonl'}
            save(directory/'summary.json',dict(execution_status='completed',protocol_sha256=protocol['manifest']['protocol_sha256'],artifact_files_sha256=hashes))
            deployment=root/'main-deployment.json'
            save(deployment,dict(project=str(project),protocol_dir=str(directory),input_sha256={str(f):sha(f) for f in directory.iterdir() if f.name!='future_test.jsonl'}))
            p=plan(ROOT);p['expected_catalog_items']=len(protocol['registries']['base_train']['id2item'])-1
            p['expected_queries']={s:len(protocol['examples'][s]) for s in ('base_train','base_val')}
            _,data,binding=load_base(deployment,p)
            self.assertEqual(set(data),{'base_train','base_val','probe'})
            self.assertEqual(binding['allowed_splits'],['base_train','base_val'])
            with (directory/'base_train.jsonl').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'input changed'):load_base(deployment,p)


@unittest.skipUnless(HAVE,'Torch/transformers unavailable; real-model checks mandatory on server')
class RealModels(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        from genrec.cf_letter_base_training import seed_all
        seed_all(2026)

    def test_sasrec_all_masked_history_is_finite_and_target_only_fitting_works(self):
        import torch
        from genrec.cf_letter_models import SASRec
        model=SASRec(32,3,dict(dim=8,heads=2,ffn=16,layers=2,dropout=.2))
        h=torch.tensor([[0,0,0],[0,1,2],[2,0,3]])
        loss=model.loss(h,torch.tensor([1,3,4]),torch.tensor([2,4,5]));loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()))
        self.assertTrue(torch.equal(model.items.weight[0],torch.zeros(8)))

    def test_actual_quantizer_gradients_frozen_features_and_reload_assignments(self):
        import torch
        from genrec.cf_letter_models import CollaborativeRQ
        model=CollaborativeRQ(torch.randn(40,8),dict(dim=8,capacity=4,books=3,commitment=.25,outer_weight=1.))
        model.initialize();before=model.features.clone()
        loss,recon,quant,_=model(model.features)
        self.assertFalse(recon.requires_grad)
        loss.backward()
        self.assertTrue(all(b.grad is not None and torch.isfinite(b.grad).all() and b.grad.norm()>0 for b in model.books))
        self.assertIsNone(model.features.grad)
        torch.optim.AdamW(model.parameters(),lr=1e-3).step()
        self.assertTrue(torch.equal(model.features,before))
        state=copy.deepcopy(model.state_dict());indices=model.assignments();model.load_state_dict(state)
        self.assertTrue(torch.equal(indices,model.assignments()))

    def test_real_gpt_cached_beam_matches_brute_force_with_holes_and_teacher_gradients(self):
        import torch
        from genrec.cf_letter_models import GenerativeRetriever,make_trie
        cfg=dict(d_model=16,layers=1,heads=2,ffn=32,dropout=.2,user_buckets=8)
        model=GenerativeRetriever(3,4,cfg);model.eval()
        mapping,_=allocate([[i//4,i%4,0] for i in range(12)],4)
        data=dict(history=np.array([[0,1,0]]),targets=np.array([2]),users=np.array([3]))
        a,m,y=gpt_arrays(data,mapping,4,8);a=torch.tensor(a);m=torch.tensor(m);y=torch.tensor(y)
        logits=model.teacher_logits(a,m,y);changed=y.clone();changed[:,-1]=(changed[:,-1]+1)%model.vocab
        self.assertTrue(torch.equal(logits,model.teacher_logits(a,m,changed)))
        direct=model.nll(a,m,y).detach();model.requires_grad_(False)
        emb=model.model.get_input_embeddings()(torch.cat([a,y],dim=1)).detach().requires_grad_(True)
        soft=model.nll(a,m,y,emb)
        self.assertTrue(torch.allclose(direct,soft,atol=2e-6));soft.sum().backward()
        self.assertTrue(torch.isfinite(emb.grad).all());self.assertGreater(float(emb.grad[:,:a.shape[1]].norm()),0)
        seq,scores=model.beam(a,m,make_trie(mapping.tolist()),len(mapping))
        all_nll=model.nll(a.expand(len(mapping),-1),m.expand(len(mapping),-1),torch.tensor(mapping))
        order=all_nll.argsort()
        self.assertTrue(torch.equal(seq[0],torch.tensor(mapping)[order]))
        self.assertTrue(torch.allclose(scores[0],-all_nll[order],atol=1e-5,rtol=1e-5))

    def test_checkpoint_sampler_optimizer_and_dropout_resume_are_exact(self):
        import torch
        from genrec.cf_letter_base_training import Sampler,rng,restore_rng
        from genrec.cf_letter_models import SASRec
        def build():
            model=SASRec(32,3,dict(dim=8,layers=1,heads=2,ffn=16,dropout=.2))
            return model,torch.optim.AdamW(model.parameters(),lr=.001),Sampler(7,3,2026)
        a,oa,sa=build();h=torch.tensor([[0,1,2]]*7);y=torch.tensor([3]*7);neg=torch.tensor([4]*7)
        def step(model,opt,sampler):
            i,_=sampler.take();opt.zero_grad();model.loss(h[i],y[i],neg[i]).backward();opt.step()
        for _ in range(2):step(a,oa,sa)
        state=copy.deepcopy((a.state_dict(),oa.state_dict(),sa.state_dict(),rng()))
        for _ in range(4):step(a,oa,sa)
        b,ob,sb=build();b.load_state_dict(state[0]);ob.load_state_dict(state[1]);sb.load_state_dict(state[2]);restore_rng(state[3])
        for _ in range(4):step(b,ob,sb)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertEqual(sa.processed,sb.processed)

    def test_tiny_actual_base_training_exports_mapping_and_checkpoint_readiness(self):
        from genrec.cf_letter_base_training import train_predictor,train_quantizer
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);p=plan(ROOT)
            p.update(history=3,batch_size=16,eval_batch_size=4,probe_size=8)
            for role in ('sasrec','gpt'):
                p[role].update(epochs=2,step_cap=4,eval_every_epochs=1,dropout=.1)
            p['sasrec'].update(dim=8,layers=1,heads=2,ffn=16)
            p['quantizer'].update(dim=8,capacity=32,batch_size=16,steps=4,monitor_every=2)
            p['gpt'].update(d_model=16,layers=1,heads=2,ffn=32,user_buckets=8)
            def data(n):
                return dict(history=np.array([[0,(i%32)+1,((i+1)%32)+1] for i in range(n)],dtype=np.int64),
                    targets=np.array([((i+2)%32)+1 for i in range(n)],dtype=np.int64),
                    users=np.arange(n,dtype=np.int64)%8,samples=np.array(['sample'+str(i) for i in range(n)]))
            values={'base_train':data(32),'base_val':data(8),'probe':np.arange(8,dtype=np.int64)}
            values['base_val']['targets'][-1]=0
            binding={'items':['item'+str(i) for i in range(32)]}
            s=train_predictor('sasrec',output,values,p,binding,'cpu')
            q=train_quantizer(output,p,binding,'cpu')
            mapping=np.load(output/'item_sids.npy',allow_pickle=False)
            g=train_predictor('gpt',output,values,p,binding,'cpu',mapping)
            self.assertEqual(s['optimizer_steps'],4);self.assertEqual(q['optimizer_steps'],4);self.assertEqual(g['optimizer_steps'],4)
            self.assertTrue(all(g['readiness'].values()))
            self.assertEqual(g['full_base_validation']['all_queries'],8)
            self.assertEqual(g['full_base_validation']['unavailable_targets'],1)


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Mandatory real Torch/GPT2 CPU checks cannot be replaced by mocks')
    unittest.main()
