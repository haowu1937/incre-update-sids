"""Exact matching, bounded Beauty audits and actual complete tiny TIGER acceptance."""
import copy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
from genrec.hungarian_sids import fit,transform,canonical
from genrec.cf_letter_protocol import read,save,sha,encoded
from genrec.tiger_hungarian_protocol import plan,cases,audit_publication,submission,NAMES,project_values
HAVE=all(importlib.util.find_spec(n) is not None for n in ['torch','transformers'])


class Pure(unittest.TestCase):
    def test_exact_permuted_tokens_recover_old_ids_including_suffix(self):
        old={str(i):[1+i//4,4+i%4] for i in range(12)};new={k:[3-(v[0]-1),4+(v[1]-4+1)%4] for k,v in old.items()}
        t=fit(old,new,[3,4]);self.assertEqual(transform(new,t),old);self.assertEqual(transform(old,t,True),new)
        self.assertTrue(all(p['after_matching_tokens']>=p['before_matching_tokens'] for p in t['positions']))

    def test_hungarian_matches_global_optimum_and_unseen_completion_is_bijective(self):
        old={str(i):[1+(i%3),4+i] for i in range(8)};new={str(i):[1+((i//2)%3),4+i] for i in range(8)}
        t=fit(old,new,[3,9]);p=t['permutations'][0]
        score=lambda perm:sum(perm[new[k][0]-1]==old[k][0]-1 for k in old)
        self.assertEqual(score(p),max(map(score,itertools.permutations(range(3)))))
        self.assertEqual(sorted(t['permutations'][1]),list(range(9)))
        bad=copy.deepcopy(t);bad['permutations'][0]=[0,0,1]
        with self.assertRaises(ValueError):transform(new,bad)

    def test_fullsid_uniqueness_collision_structure_and_common_items_only(self):
        old={'a':[1,3,5,7],'b':[1,3,5,8],'c':[2,4,6,7]};new=copy.deepcopy(old);new['new']=[2,4,6,8]
        t=fit(old,new,[2]*4);self.assertEqual(t['common_items'],3);self.assertEqual(t['common_items_sha256'],canonical(['a','b','c']))
        self.assertEqual(transform(transform(new,t),t,True),new)
        with self.assertRaises(ValueError):fit({'a':[0,3,5,7]},new,[2]*4)

    def test_beauty_size_audit_is_local_full_and_bounded_digest_only(self):
        full=dict(input_sha256={('/home/hao/frec/graph-informed-sids-web/'+str(i)*150):'a'*64 for i in range(400)},protocol_sha256='a'*64,probe_sample_ids=['b'*64+str(i) for i in range(512)],train_sample_ids_sha256='c'*64,fit_catalog='old Base train only',source_acceptance={'hash':'d'*64})
        self.assertGreater(len(encoded(full)),65536)
        with tempfile.TemporaryDirectory() as tmp:
            public=audit_publication(tmp,full);self.assertLess(len(encoded(public)),4096);self.assertEqual(read(Path(tmp)/'audit_full.json'),full);self.assertNotIn('input_sha256',public)

    def test_new_6000_budget_reuses_one_control_and_installed_generic_contract(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'scripts/lab_buffer'));import task_runtime
        p=plan();self.assertEqual(sum(c['steps'] for c in cases(p)),6000);self.assertNotIn('historical_unmapped',[c['name'] for c in cases(p)])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'task-runtime-policy.json',dict(protocol_version=1,max_parallel=2,read_roots=[str(root)]));b=submission('a'*40,ROOT,root)
            runner=SimpleNamespace(root=root,config=dict(task_runtime_version=1,worker_id='server2'),read=lambda code,path:(ROOT/path).read_bytes() if (ROOT/path).is_file() else None)
            task_runtime.validate(runner,b['request'],encoded(b['request']),b['contract'],b['scope']);self.assertEqual(len(b['request']['artifacts']),8)

    def test_exact_pause_archival_preserves_failed_training_and_refuses_other_reason(self):
        import hashlib
        from types import SimpleNamespace
        spec=importlib.util.spec_from_file_location('reconcile_hung',ROOT/'scripts/lab_buffer/reconcile_tiger_hungarian.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        for reason,allowed in [(helper.REASON,True),('RuntimeError: unrelated network error',False)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'state').mkdir();(root/'bin').mkdir();code=root/'code';(code/'docs/experiments').mkdir(parents=True)
                old=dict(job_id=helper.SOURCE,status='failed',code_sha='b'*40,finished_at='2026-11-02T11:30:00+00:00')
                save(root/'state'/(helper.SOURCE+'.json'),dict(result=old,pending_upload=False));compact=root/'workspaces'/helper.SOURCE/'outputs/cf-gpt-lr/compact.json';save(compact,dict(optimizer_steps_spent=10000))
                p=plan();p['previous_failure']=dict(code_sha=old['code_sha'],result_sha256=hashlib.sha256(encoded(old)).hexdigest(),compact_sha256=sha(compact));manifest=encoded(p);(code/'docs/experiments/tiger_hungarian.json').write_bytes(manifest)
                request=dict(job_id=helper.JOB,code_sha='a'*40);request_raw=encoded(request);state=dict(lease=None,pause_requested=False,mode='queued',active_job_id=helper.JOB,code_sha='a'*40,request_sha256=hashlib.sha256(request_raw).hexdigest(),receipts={helper.SOURCE:{'status':'needs_attention'}})
                save(root/'PAUSED.json',dict(reason=reason,time='2026-11-02T11:31:00+00:00'));save(root/'main-service-error.json',dict(error=reason,new_jobs_paused=True))
                class Runner:
                    def __init__(self,path):self.env={};self.config={k:k for k in ('code_branch','requests_branch','results_branch')}
                    def fetch(self):pass
                    def ref(self,key):return key
                    def git(self,*a,**kw):return SimpleNamespace(returncode=0)
                    def read(self,ref,path):return manifest if path.endswith('tiger_hungarian.json') else request_raw if path.startswith('jobs/') else encoded(state)
                runner=SimpleNamespace(Runner=Runner,FINAL={'failed','completed'},atomic_json=save,now=lambda:'fixture')
                task=SimpleNamespace(load_contract=lambda *a:({'optimizer_step_reservation':6000},None,{'max_optimizer_steps':6000,'future_passes_new':3}))
                before=sha(compact)
                with patch.object(helper,'ROOT',code),patch.dict(sys.modules,{'runner':runner,'task_runtime':task}):
                    if allowed:
                        value=helper.reconcile(root,'a'*40);self.assertFalse(value['task_launched']);self.assertFalse(value['runner_upgraded']);self.assertFalse((root/'PAUSED.json').exists());self.assertEqual(len(list((root/'resolved-pauses').glob('*/PAUSED.json'))),1)
                    else:
                        with self.assertRaisesRegex(ValueError,'not the exact'):helper.reconcile(root,'a'*40)
                        self.assertTrue((root/'PAUSED.json').exists())
                self.assertEqual(sha(compact),before)


@unittest.skipUnless(HAVE,'Mandatory actual RQ/TIGER fixture and complete scientific verifier run on server')
class Actual(unittest.TestCase):
    def test_complete_four_condition_pipeline_sampling_prediction_audit_and_verifier(self):
        import torch
        from genrec import future_evaluation as f
        import evaluate_historical_future as old
        from genrec import tiger_hungarian_experiment as e
        from genrec.tiger_hungarian_protocol import projections
        from run_cf_letter_downstream import counters
        from run_cf_letter_multiseed import seal
        import verify_tiger_hungarian as verifier
        spec=importlib.util.spec_from_file_location('hung_tiger_fixture',ROOT/'tests/incremental/test_historical_future.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        torch.set_num_threads(1);klass=fixture.RealModels;klass.setUpClass()
        try:
            root=klass.root;previous=root/'previous_future'
            args=old.parser().parse_args(['--e1-dir',str(root/'e1'),'--retuned-dir',str(root/'retuned'),'--output',str(previous),'--device','cpu','--max-steps','2','--warmup-steps','2','--selection-steps','1','2','--num-beams','4','--topk','1','3','--probe-size','10','--eval-batch-size','3','--save-every','1'])
            report=old.run(args);p=plan();p.update(steps_per_arm=2,optimizer_step_cap=6,selection_steps=[0,1,2],bootstrap_replicates=10,fixture_previous=report,source_acceptance={'fixture':True})
            deployment=root/'main-deployment.json';save(deployment,dict(runner_root=str(root),e1_dir=str(root/'e1'),retuned_dir=str(root/'retuned'),future_dir=str(previous)))
            original={str(path):sha(path) for path in root.rglob('*') if path.is_file()}
            workspace=root/'workspaces'/p['job_id'];output=workspace/'outputs/tiger-hungarian';output.mkdir(parents=True)
            for n in NAMES:save(output/(n+'.json'),{})
            e.prepare(deployment,output,p,torch.device('cpu'),fixture=True)
            load_stage=f.load_stage
            def no_future(ex,stage,*a,**kw):
                if stage=='future_test':raise AssertionError('Future opened during fitting')
                return load_stage(ex,stage,*a,**kw)
            with patch.object(f,'load_stage',side_effect=no_future):
                for name in p['new_training_arms']:
                    (output/'arms'/name).mkdir(parents=True);e.train(deployment,output,p,name,torch.device('cpu'))
            save(output/'launch_journal.json',cases(p));lock={}
            for name in p['arms']:
                folder=output/('reused' if name=='historical_unmapped' else 'arms')/name;s=read(folder/'summary.json');lock[name]=dict(selected_sha256=sha(folder/'best.pth'),selected_step=s['best_update_optimizer_step'],mapping_sha256=sha(output/'maps'/(name+'_deployment.json')))
            save(output/'selection_locked.json',dict(arms=lock,future_used_for_selection=False,selection_split='update_val'))
            for name in p['new_training_arms']:e.evaluate_future(deployment,output,p,name,torch.device('cpu'))
            projections(output,p);save(output/'comparisons.json',e.comparisons(output,p));total,counts,intents=counters(output)
            compact=dict(execution_status='completed',job_id=p['job_id'],code_sha='fixture',future_passes_new=3,original_inputs_unchanged=True,selection_locked_before_new_future=True,rqvae_training_performed=False,base_tiger_training_performed=False,holdout_evaluated=False,future_used_for_matching_or_selection=False,optimizer_steps_spent=total,stage_step_counts=counts,uncertain_step_records=intents,local_artifact_index_sha256=seal(output),returned_artifact_sha256={n:sha(output/(n+'.json')) for n in NAMES if n!='compact'})
            save(output/'compact.json',compact);context=root/'verify-context.json';save(context,dict(workspace=str(workspace),request={'code_sha':'fixture'},contract={'input_sha256':{str(deployment):sha(deployment)}},contract_sha256='fixture'))
            before={str(path):sha(path) for path in output.rglob('*') if path.is_file()};value=verifier.verify(context,configuration=p,fixture=True)
            self.assertEqual(value['actual_optimizer_steps'],6);self.assertIn('Recall@3',value['report_markdown']);self.assertTrue(all(sha(path)==h for path,h in before.items()));self.assertTrue(all(sha(path)==h for path,h in original.items()))
            self.assertEqual(set(project_values(output,p)[1]),set(p['arms']))
        finally:klass.tearDownClass()


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Real TIGER/Hungarian/full acceptance preflight required')
    unittest.main()
