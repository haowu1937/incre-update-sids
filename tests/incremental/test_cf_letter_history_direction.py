"""Training-only scope tests and actual GPT/codebook STE direction checks."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
import numpy as np
from genrec.cf_letter_protocol import allocate,encoded,read,save,sha
from genrec.cf_letter_history_protocol import cases,plan,submission,transfer_refresh

HAVE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


class Protocol(unittest.TestCase):
    def test_scope_matrix_budget_and_generic_interface(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'scripts/lab_buffer'));import task_runtime
        p=plan(ROOT);matrix=cases(p);self.assertEqual(len(matrix),12);self.assertEqual(sum(c['steps'] for c in matrix),60)
        self.assertEqual(len(set(c['name'] for c in matrix)),12);self.assertFalse(p['lambda_sweep_authorized']);self.assertFalse(p['validation_evaluation'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'task-runtime-policy.json',dict(protocol_version=1,max_parallel=2,read_roots=[str(root)]))
            unapproved=root/'unapproved';pending=copy.deepcopy(p);pending['execution_authorized']=False;save(unapproved/'docs/experiments/cf_letter_history_direction.json',pending)
            with self.assertRaisesRegex(ValueError,'no explicit execution authorization'):submission('a'*40,'b'*64,root,unapproved)
            import run_cf_letter_history_direction as driver
            with self.assertRaisesRegex(ValueError,'no explicit execution authorization'):driver.child(root,pending)
            with self.assertRaisesRegex(ValueError,'no explicit execution authorization'):driver.run(root/'absent.json',root/'absent-output',pending)
            # The only authorized manifest here is an isolated synthetic contract fixture.
            proposal=root/'fixture';p['execution_authorized']=True;save(proposal/'docs/experiments/cf_letter_history_direction.json',p)
            def source(code,path):
                chosen=proposal/path if path=='docs/experiments/cf_letter_history_direction.json' else ROOT/path
                return chosen.read_bytes() if chosen.is_file() else None
            r=SimpleNamespace(root=root,config=dict(task_runtime_version=1,worker_id='server2'),read=source)
            v=submission('a'*40,'b'*64,root,proposal);task_runtime.validate(r,v['request'],encoded(v['request']),v['contract'],v['scope'])
            self.assertEqual(v['contract']['optimizer_step_reservation'],60)

    def test_training_loader_can_exclude_all_validation_queries(self):
        # Reuse the existing causal/poison-Future fixture with no validation file access.
        from genrec.temporal import build_temporal_protocol
        from genrec.cf_letter_refresh_protocol import anchor_indices,load_data
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);project=root/'project';directory=project/'cache/incremental/Beauty/beauty_20261007T104452Z';directory.mkdir(parents=True)
            protocol=build_temporal_protocol([('u'+str(i%8),'item'+str(i%53),1000000000+i,None) for i in range(800)])
            save(directory/'protocol_manifest.json',protocol['manifest']);save(directory/'registries.json',protocol['registries'])
            names=['protocol_manifest.json','registries.json']
            for split in ('base_train','update_train'):
                name=split+'.jsonl';names.append(name);(directory/name).write_text(''.join(json.dumps(r)+'\n' for r in protocol['examples'][split]))
            for split in ('base_val','update_val','future_test'):(directory/(split+'.jsonl')).write_text('MUST NOT READ')
            save(directory/'summary.json',dict(execution_status='completed',protocol_sha256=protocol['manifest']['protocol_sha256'],artifact_files_sha256={n:sha(directory/n) for n in names}))
            p=plan(ROOT);p['old_catalog_items']=len(protocol['registries']['base_train']['id2item'])-1;p['anchor_size']=8;p['expected_queries']={s:len(protocol['examples'][s]) for s in p['allowed_query_splits']}
            e1=project/'e1';e1.mkdir();rows=protocol['examples']['base_train'];indices=anchor_indices(rows,8,2026)
            (e1/'anchor_train.jsonl').write_text(''.join(json.dumps(dict(source_row=int(i),**rows[i]))+'\n' for i in indices))
            paths=[directory/n for n in names+['summary.json']]+[e1/'anchor_train.jsonl'];deployment=root/'main-deployment.json'
            save(deployment,dict(project=str(project),protocol_dir=str(directory),e1_dir=str(e1),input_sha256={str(f):sha(f) for f in paths}))
            data,binding=load_data(deployment,p);self.assertIsNone(data['val']);self.assertEqual(binding['probe_sample_ids'],[])
            self.assertFalse(any('/update_val.jsonl' in name or '/future_test.jsonl' in name for name in binding['input_sha256']))

    def test_read_only_accepted_refresh_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=plan(ROOT);source=root/'workspaces'/p['refresh_predecessor']['job_id']/'outputs/cf-refresh';source.mkdir(parents=True)
            (source/'cf.npy').write_bytes(b'fixed E1');save(source/'compact.json',dict(research_outcome='feature_refresh_measured',code_sha=p['refresh_predecessor']['code_sha']))
            p['refresh_predecessor'].update(files_sha256={'cf.npy':sha(source/'cf.npy')},compact_sha256=sha(source/'compact.json'))
            bound=transfer_refresh(root,root/'new-output',p);self.assertEqual(len(bound),2)
            (source/'cf.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):transfer_refresh(root,root/'another-output',p)


@unittest.skipUnless(HAVE,'Torch/transformers unavailable; mandatory actual-model server checks')
class RealModels(unittest.TestCase):
    def setUp(self):
        import torch
        from genrec.cf_letter_base_training import seed_all
        torch.set_num_threads(1);seed_all(2026)

    def fixture(self,output):
        import torch
        from genrec.cf_letter_models import CollaborativeRQ,GenerativeRetriever
        from genrec.cf_letter_refresh_training import paired_nll,quantize,records
        from genrec.cf_letter_refresh_protocol import versioned_mapping,damage
        p=plan(ROOT);p.update(history=3,fixed_anchor_queries=8,gradient_probe_batches=2,content_batch_size=16,anchor_size=16,eval_batch_size=4,bootstrap_replicates=10,old_catalog_items=48)
        p['gpt'].update(d_model=16,layers=1,heads=2,ffn=32,user_buckets=8);p['quantizer'].update(dim=8,capacity=16)
        p['predecessor'].update(code_sha='test',gpt_step=2)
        base=output/'inputs/base';refresh=output/'inputs/refresh';base.mkdir(parents=True);(refresh/'feature_only').mkdir(parents=True);(refresh/'baseline').mkdir()
        features=torch.randn(48,8);q=CollaborativeRQ(features,p['quantizer']);q.initialize();codes=q.assignments().numpy();mapping,_=allocate(codes,16)
        model=GenerativeRetriever(3,16,p['gpt']);model.eval();model.requires_grad_(False)
        anchors=dict(history=np.array([[0,i+1,i+2] for i in range(16)]),targets=np.arange(3,19),users=np.arange(16)%8,samples=np.asarray(['sample'+str(i) for i in range(16)]))
        data=dict(anchors=anchors,anchor_users=np.asarray(['user'+str(i%8) for i in range(16)]),anchor_source_rows=np.arange(16))
        fixed=torch.randperm(16,generator=torch.Generator().manual_seed(p['seed']+77)).numpy()[:8]
        for scale in (.05,.2,.5,1.):
            new=torch.cat([features+scale*torch.randn_like(features),torch.randn(4,8)]);newcodes,_=quantize(q,new[:48],'cpu');newmap,capacity=versioned_mapping(mapping,newcodes,16)
            if newmap is None:continue
            a,b,changed=paired_nll(model,anchors,mapping,newmap,p,'cpu')
            if np.maximum(b-a,0).mean()>1e-4 and np.maximum((b-a)[fixed],0).mean()>1e-4:break
        else:raise AssertionError('Synthetic real-model fixture must measure positive hard H')
        np.save(base/'cf_embeddings.npy',features.numpy());np.save(base/'item_sids.npy',mapping)
        for name,source in [('gpt',model),('quantizer',q)]:
            (base/name).mkdir();torch.save(dict(model=source.state_dict(),step=2,code_sha='test'),base/name/'selected.pth')
        np.save(refresh/'refreshed_cf_embeddings.npy',new.numpy());np.save(refresh/'feature_only/semantic_codes.npy',newcodes);np.save(refresh/'feature_only/item_sids.npy',newmap);np.save(refresh/'baseline/old_anchor_nll.npy',a)
        p['refresh_predecessor']['expected_initial_full_H']=damage(records(data,a,b,changed),10,2026)['H']
        binding=dict(items=['item'+str(i) for i in range(52)],old_items=['item'+str(i) for i in range(48)])
        return p,data,binding

    def test_actual_ste_hard_forward_all_book_gradients_and_frozen_inputs(self):
        import torch
        from genrec.cf_letter_history_direction import Engine
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);p,data,binding=self.fixture(output);engine=Engine(output,data,binding,p,'cpu')
            loss,_,_,_=engine.historical(engine.fixed,engine.initial_codes,engine.initial_mapping)
            gradients=torch.autograd.grad(loss,list(engine.model.books))
            self.assertTrue(all(torch.isfinite(g).all() and g.norm()>0 for g in gradients))
            self.assertTrue(all(v.grad is None and not v.requires_grad for v in engine.teacher.parameters()))
            self.assertFalse(engine.model.features.requires_grad);self.assertIsNone(engine.model.features.grad)
            calibration=engine.calibrate();self.assertGreater(calibration['history_weight'],0)

    def test_actual_all_branches_share_start_sampling_and_prospective_counter(self):
        from genrec.cf_letter_history_direction import Engine
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);p,data,binding=self.fixture(output);inputs={str(path):sha(path) for path in (output/'inputs').rglob('*') if path.is_file()}
            engine=Engine(output,data,binding,p,'cpu');result=engine.run()
            self.assertEqual(len(result['branches']),12);self.assertEqual(result['optimizer_steps_spent'],sum(b['optimizer_steps'] for b in result['branches']))
            self.assertLessEqual(result['optimizer_steps_spent'],60);self.assertTrue(result['teacher_unchanged']);self.assertTrue(result['cf_embeddings_unchanged'])
            self.assertEqual(inputs,{name:sha(name) for name in inputs})
            for name in ('sgd_curves','adamw_curves'):self.assertLess((output/(name+'.json')).stat().st_size,65536)


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Actual Torch/GPT/codebook checks are mandatory before Beauty micro-updates')
    unittest.main()
