"""Exact scope, fixed cohorts, seed confidence intervals and real two-seed pipeline."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
import numpy as np
from genrec.cf_letter_protocol import read,save,sha,encoded
from genrec.cf_letter_multiseed_protocol import JOB,SCOPE,MANIFEST,ARTIFACTS,plan,settings,rq_cases,gpt_cases,comparison_pairs,summarize,interval,submission
HAVE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


class Protocol(unittest.TestCase):
    def test_exact_scope_and_existing_generic_contract(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'scripts/lab_buffer'));import task_runtime
        p=plan(ROOT);self.assertEqual(sum(c['steps'] for c in rq_cases(p)),10000);self.assertEqual(sum(c['steps'] for c in gpt_cases(p)),30000)
        self.assertTrue(all(c['beta1']==.9 for c in rq_cases(p)));self.assertEqual(len(gpt_cases(p)),15)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'task-runtime-policy.json',dict(protocol_version=1,max_parallel=2,read_roots=[str(root)]))
            runner=SimpleNamespace(root=root,config=dict(task_runtime_version=1,worker_id='server2'),read=lambda code,path:(ROOT/path).read_bytes() if (ROOT/path).is_file() else None)
            b=submission('a'*40,'b'*64,root,ROOT);task_runtime.validate(runner,b['request'],encoded(b['request']),b['contract'],b['scope'])
            self.assertEqual(len(b['request']['artifacts']),8);self.assertEqual(b['contract']['optimizer_step_reservation'],40000)
            for key,value in [('seeds',[2026,2027,2028,2029,2030]),('optimizer_step_cap',40001),('rq_steps',999),('parallel',3),('retry',True)]:
                q=copy.deepcopy(p);q[key]=value;save(root/MANIFEST,q)
                with self.assertRaises(ValueError):plan(root)

    def test_training_seeds_are_separate_from_fixed_cohort_and_probe(self):
        p=plan(ROOT)
        self.assertEqual(settings(ROOT,p,'rq')['seed'],2026);self.assertEqual(settings(ROOT,p,'gpt')['seed'],2026)
        for seed in p['seeds']:
            self.assertEqual(settings(ROOT,p,'rq',seed)['seed'],seed)
            self.assertEqual(settings(ROOT,p,'gpt',seed)['seed'],seed)
        # Existing poisoned-Future temporal fixture exercises the same production loader.
        spec=importlib.util.spec_from_file_location('multiseed_temporal',ROOT/'tests/incremental/test_cf_letter_downstream.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        from genrec.cf_letter_downstream_protocol import load_training,load_stage
        with tempfile.TemporaryDirectory() as tmp:
            q,deployment,directory=fixture.temporal_fixture(Path(tmp));q.update(job_id=JOB,arms=[{k:v for k,v in c.items() if k not in ('steps','phase')} for c in gpt_cases(p)])
            (directory/'future_test.jsonl').write_text('MUST NOT OPEN')
            _,_,probe,a=load_training(deployment,q);self.assertEqual(len(probe),4)
            with self.assertRaisesRegex(ValueError,'locked'):load_stage(deployment,q,'future_test')
            self.assertFalse(any('future' in f for f in a['input_sha256']))

    def test_all_fifteen_models_required_before_future(self):
        from genrec.cf_letter_downstream_protocol import valid_lock
        p=plan(ROOT);q=settings(ROOT,p,'gpt')
        lock=dict(job_id=JOB,future_used_for_selection=False,arms={c['name']:dict(selected_step=250,selected_sha256='a'*64,mapping_sha256='b'*64) for c in gpt_cases(p)})
        valid_lock(lock,q);lock['arms'].pop('s2031_historical')
        with self.assertRaises(ValueError):valid_lock(lock,q)

    def test_seed_intervals_are_paired_and_reject_missing_or_nonfinite_seeds(self):
        p=plan(ROOT);future={}
        for c in gpt_cases(p):
            value=.01+(c['seed']-2027)*.001+{'ft_old':0,'plain':.001,'historical':.002}[c['method']]
            future[c['name']]={d:{f'{m}@{k}':value for m in ('recall','ndcg') for k in (5,10,20)} for d in ('full_denominator','conditional_on_servable')}
        r=summarize(future,p,[5,10,20]);v=r['paired']['full_denominator']['historical_minus_plain']['ndcg@10']
        self.assertAlmostEqual(v['mean'],.001);self.assertLess(v['sd'],1e-15)
        single=interval([1,2,3,4,5]);self.assertEqual(single['mean'],3);self.assertAlmostEqual(single['sd'],np.std([1,2,3,4,5],ddof=1))
        self.assertAlmostEqual(single['ci95'][1]-3,2.7764451051977987*np.std([1,2,3,4,5],ddof=1)/np.sqrt(5))
        with self.assertRaises(ValueError):interval([1,float('nan')])
        future.pop('s2031_historical')
        with self.assertRaises(ValueError):summarize(future,p,[5,10,20])

    def test_nested_phase_counters_include_uncertain_intents(self):
        from run_cf_letter_multiseed import total_counters
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'rq/arms/s2027_plain/s2027_plain/optimizer_steps.json',dict(optimizer_steps_completed=999,optimizer_step_intent=1000))
            save(root/'gpt/arms/s2027_ft_old/optimizer_steps.json',dict(optimizer_steps_completed=2000,optimizer_step_intent=None))
            n,steps,intents=total_counters(root);self.assertEqual(n,3000);self.assertEqual(len(intents),1);self.assertEqual(steps['rq/s2027_plain'],1000)


@unittest.skipUnless(HAVE,'Actual multi-seed Torch/GPT/RQ checks mandatory on server')
class RealModels(unittest.TestCase):
    def test_actual_two_seed_RQ_to_GPT_global_lock_future_and_scientific_acceptance(self):
        import torch
        from genrec.cf_letter_base_training import seed_all
        from genrec.cf_letter_optimizer_training import OptimizerEngine
        from genrec.cf_letter_optimizer_protocol import save_anchors
        from genrec.cf_letter_downstream_protocol import save_arrays,paired,digest
        from genrec.cf_letter_downstream_training import train_arm,future_arm
        from run_cf_letter_multiseed import seal,total_counters
        import verify_cf_letter_multiseed as verifier
        torch.set_num_threads(1);seed_all(2026)
        spec=importlib.util.spec_from_file_location('multiseed_rq_fixture',ROOT/'tests/incremental/test_cf_letter_optimizer.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'outputs/cf-multiseed';rq=output/'rq';gpt=output/'gpt';rq.mkdir(parents=True);gpt.mkdir()
            qr,data,binding=fixture.RealModels().fixture(rq);qr.update(steps_per_branch=4,optimizer_step_cap=16)
            # Measured positive-H synthetic anchors keep every new seed's fixed
            # monitor away from an accidental zero-H fixture; no production guard changes.
            from genrec.cf_letter_refresh_training import teacher,paired_nll
            model,_=teacher(rq,qr,'cpu');old_map=np.load(rq/'inputs/base/item_sids.npy');new_map=np.load(rq/'inputs/refresh/feature_only/item_sids.npy')
            old,new,_=paired_nll(model,data['anchors'],old_map,new_map,qr,'cpu');positive=np.flatnonzero(new-old>1e-4)
            self.assertGreater(len(positive),0);take=np.resize(positive,qr['anchor_size'])
            for key in ('history','targets','users'):data['anchors'][key]=data['anchors'][key][take].copy()
            old,new,_=paired_nll(model,data['anchors'],old_map,new_map,qr,'cpu')
            np.save(rq/'inputs/refresh/baseline/old_anchor_nll.npy',old);qr['refresh_predecessor']['expected_initial_full_H']=float(np.maximum(new-old,0).mean())
            del model
            p=plan(ROOT);p.update(seeds=[2027,2028],rq_steps=4,gpt_steps=4,optimizer_step_cap=40,future_passes=6)
            save(rq/'inputs/base/compact.json',{});save(rq/'inputs/base/data_audit.json',{});save(rq/'inputs/refresh/compact.json',{})
            for folder,key in [('base','predecessor'),('refresh','refresh_predecessor')]:
                qr[key]['files_sha256']={str(f.relative_to(rq/'inputs'/folder)):sha(f) for f in (rq/'inputs'/folder).rglob('*') if f.is_file() and f.name not in ('compact.json','data_audit.json')}
                qr[key]['compact_sha256']=sha(rq/'inputs'/folder/'compact.json')
            qr['predecessor']['audit_sha256']=sha(rq/'inputs/base/data_audit.json');qr['audit_predecessor']={'files_sha256':{}}
            save_anchors(rq/'prepared.npz',data);save(rq/'binding.json',binding);save(rq/'launch_journal.json',rq_cases(p));rqresults={}
            for case in rq_cases(p):
                directory=rq/'arms'/case['name'];directory.mkdir(parents=True);shutil.copytree(rq/'inputs',directory/'inputs');shutil.copyfile(rq/'prepared.npz',directory/'prepared.npz')
                params=copy.deepcopy(qr);params['seed']=case['seed'];seed_all(case['seed']);engine=OptimizerEngine(directory,data,binding,params,'cpu')
                value=engine.run_case(case);value.update(code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'),case=case);save(directory/'arm_result.json',value);rqresults[case['name']]=value
            save(output/'rq_results.json',rqresults);shutil.copytree(rq/'inputs',gpt/'inputs')
            qg=copy.deepcopy(qr);qg.update(job_id=JOB,scope_id=SCOPE,seed=2026,steps_per_arm=4,optimizer_step_cap=24,batch_size=4,lr=1e-4,weight_decay=.01,betas=[.9,.999],eps=1e-8,max_grad_norm=1.,
                selection_steps=[0,1,2,4],selection_metric='ndcg@10',topk=[5,10,20],beam=20,arms=[{k:v for k,v in c.items() if k not in ('steps','phase')} for c in gpt_cases(p)])
            qg.pop('zero_step_repair_of',None)
            def arrays(prefix,n):
                a={k:np.resize(v,(n,*v.shape[1:])) for k,v in data['anchors'].items()};a['samples']=np.asarray([prefix+str(i) for i in range(n)])
                a['raw_users']=np.asarray(['user'+str(i%8) for i in range(n)]);a['source_rows']=np.arange(n);return a
            train=arrays('t',16);val=arrays('v',8);future=arrays('f',24);future['targets'][-1]=0;probe=np.arange(4)
            b=dict(input_sha256={},all_update_train_queries=16,eligible_update_train_queries=16,excluded_update_train_queries=0,probe_sample_ids=val['samples'][probe].tolist(),catalog_items=48,training_sample_ids_sha256=digest(train['samples'].tolist()))
            save(gpt/'data_audit.json',b);save_arrays(gpt/'prepared.npz',{**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()},'probe':probe});save_arrays(gpt/'future_prepared.npz',future)
            (gpt/'mappings').mkdir();training={}
            for case in gpt_cases(p):
                source=rq/'inputs/base/item_sids.npy' if case['source'] is None else rq/'arms'/case['source']/case['source']/'item_sids_4.npy'
                shutil.copyfile(source,gpt/'mappings'/(case['name']+'.npy'));params=copy.deepcopy(qg);params['seed']=case['seed']
                training[case['name']]=train_arm(gpt,case,train,val,probe,params,'cpu',lambda:None)
            save(gpt/'training.json',training);save(gpt/'launch_journal.json',gpt_cases(p))
            lock=dict(job_id=JOB,code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'),created_at=time.time(),future_used_for_selection=False,probe_sample_ids=b['probe_sample_ids'],
                arms={n:dict(selected_step=v['selected_update_step'],selected_sha256=v['selected_sha256'],mapping_sha256=v['mapping_sha256']) for n,v in training.items()})
            save(gpt/'selection_locked.json',lock);lock_sha=sha(gpt/'selection_locked.json');save(gpt/'future_opened.json',dict(time=time.time(),selection_lock_sha256=lock_sha))
            futures={c['name']:future_arm(gpt,c,future,qg,'cpu',lock_sha,lambda:None) for c in gpt_cases(p)};save(gpt/'future.json',futures);save(output/'future.json',futures)
            rows={n:[json.loads(v) for v in (gpt/'arms'/n/'future_queries.jsonl').read_text().splitlines()] for n in futures}
            pairs={a+'_minus_'+b:paired(rows[b],rows[a],qg) for b,a in comparison_pairs(p)};save(gpt/'comparisons.json',pairs)
            stats=summarize(futures,p,qg['topk']);stats['per_seed_user_cluster_comparisons']=pairs;save(output/'statistics.json',stats)
            save(output/'runs.json',{n:{k:v for k,v in d.items() if k!='curve'} for n,d in training.items()})
            save(output/'selections.json',dict(arms={n:dict(selected_update_step=d['selected_update_step'],selected_sha256=d['selected_sha256'],mapping_sha256=d['mapping_sha256'],probe_curve=[dict(step=v['step'],ndcg10=v['metrics']['full_denominator']['ndcg@10']) for v in d['curve']]) for n,d in training.items()},lock_sha256=lock_sha))
            save(output/'data_audit.json',dict(input_sha256={},cohort_seed=2026,rq_prepared_sha256=sha(rq/'prepared.npz'),gpt_data_audit_sha256=sha(gpt/'data_audit.json')))
            from run_cf_letter_downstream import counters
            n,counts,intents=counters(gpt);save(gpt/'compact.json',dict(stage=SCOPE,job_id=JOB,code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'),execution_status='completed',original_inputs_unchanged=True,
                selection_locked_before_future=True,future_used_for_selection=False,quantizer_training_performed=False,sasrec_training_performed=False,holdout_evaluated=False,returned_artifact_sha256={},local_artifact_index_sha256=seal(gpt),selection_lock_sha256=lock_sha,optimizer_steps_spent=n,stage_step_counts=counts,uncertain_step_records=intents))
            n,counts,intents=total_counters(output);save(output/'compact.json',dict(stage=SCOPE,job_id=JOB,code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'),execution_status='completed',seeds=p['seeds'],
                original_inputs_unchanged=True,selection_locked_before_future=True,future_used_for_selection=False,sasrec_training_performed=False,base_training_performed=False,holdout_evaluated=False,
                stop_new_stages_reason=None,future_names=list(futures),optimizer_steps_spent=n,stage_step_counts=counts,uncertain_step_records=intents,local_artifact_index_sha256=seal(output),returned_artifact_sha256={name:sha(output/(name+'.json')) for name in ARTIFACTS if name!='compact'}))
            context=root/'context.json';save(context,dict(workspace=str(root),request={'code_sha':os.environ.get('LAB_BUFFER_CODE_SHA')},contract={'input_sha256':{'/fixture/main-deployment.json':'x'}},contract_sha256='fixture'))
            def configured(root,p,stage,seed=None):
                q=copy.deepcopy(qr if stage=='rq' else qg);q['seed']=2026 if seed is None else seed;return q
            before={str(f):sha(f) for f in output.rglob('*') if f.is_file()}
            with patch.object(verifier,'settings',side_effect=configured),patch.object(verifier,'load_data',return_value=(data,binding)),patch('verify_cf_letter_downstream.load_training',return_value=(train,val,probe,b)),patch('verify_cf_letter_downstream.load_stage',return_value=(future,{})):
                verdict=verifier.verify(context,assignment_device='cpu',configuration=p)
            self.assertEqual(verdict['status'],'verified');self.assertEqual(verdict['actual_optimizer_steps'],40)
            self.assertTrue(all(sha(f)==v for f,v in before.items()))
            self.assertEqual(len(futures),6);self.assertEqual(list(stats['groups']['full_denominator']),['ft_old','plain','historical'])


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Actual multi-seed RQ/GPT/lock/Future/full-verifier checks required before Beauty')
    unittest.main()
