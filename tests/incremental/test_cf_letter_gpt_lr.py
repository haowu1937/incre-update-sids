"""LR-only contract, no Future reads, prefix reuse and actual CPU acceptance."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
import numpy as np
from genrec.cf_letter_protocol import read,save,sha,encoded,allocate
from genrec.cf_letter_gpt_lr_protocol import JOB,SCOPE,plan,cases,settings,submission,source_files,baseline_result,comparisons,publications,transfer
HAVE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


class Protocol(unittest.TestCase):
    def test_exact_lr_only_scope_uses_generic_runtime_without_host_changes(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'scripts/lab_buffer'));import task_runtime
        p=plan(ROOT);self.assertEqual(len(cases(p)),10);self.assertEqual(sum(c['steps'] for c in cases(p)),10000)
        q=settings(ROOT,p);old=settings(ROOT,p,baseline=True)
        self.assertEqual(q['lr'],3e-5);self.assertEqual(old['lr'],1e-4)
        self.assertEqual({k:v for k,v in q.items() if k!='lr'},{k:v for k,v in old.items() if k!='lr'})
        self.assertEqual(q['selection_steps'],[0,100,250,500,1000]);self.assertEqual(q['seed'],2026)
        self.assertEqual(settings(ROOT,p,2031)['seed'],2031)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'task-runtime-policy.json',dict(protocol_version=1,max_parallel=2,read_roots=[str(root)]))
            runner=SimpleNamespace(root=root,config=dict(task_runtime_version=1,worker_id='server2'),read=lambda code,path:(ROOT/path).read_bytes() if (ROOT/path).is_file() else None)
            b=submission('a'*40,'b'*64,root,ROOT);task_runtime.validate(runner,b['request'],encoded(b['request']),b['contract'],b['scope'])
            self.assertEqual(b['scope']['future_passes'],0);self.assertEqual(len(b['request']['artifacts']),7)
            self.assertEqual(b['contract']['optimizer_step_reservation'],10000)

    def test_source_whitelist_excludes_future_holdout_and_RQ_weights(self):
        files=source_files(plan(ROOT))
        self.assertFalse(any(any(k in f for k in ('future','holdout','rq/arms')) for f in files))
        self.assertFalse(any('candidate_2000' in f for f in files))
        self.assertEqual(sum('/mappings/' in f for f in files),10)
        self.assertTrue(all(any(c['name'] in f for c in cases(plan(ROOT))) or '/inputs/base/' in f for f in files))

    def test_training_loader_keeps_fixed_probe_and_does_not_open_future(self):
        spec=importlib.util.spec_from_file_location('lr_temporal',ROOT/'tests/incremental/test_cf_letter_downstream.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        from genrec.cf_letter_downstream_protocol import load_training
        with tempfile.TemporaryDirectory() as tmp:
            q,d,directory=fixture.temporal_fixture(Path(tmp));(directory/'future_test.jsonl').write_text('MUST NOT OPEN')
            _,_,probe,b=load_training(d,q);self.assertEqual(len(probe),4);self.assertFalse(any('future' in f for f in b['input_sha256']))

    def test_seed_differences_are_paired_and_no_success_direction_is_required(self):
        p=plan(ROOT);baseline={};training={}
        for c in cases(p):
            v=.01+(c['seed']-2027)*.001+(.002 if c['method']=='historical' else 0)
            metrics={f'{m}@{k}':v for m in ('recall','ndcg') for k in p['topk']}
            baseline[c['name']]={cohort:{d:metrics for d in ('full_denominator','conditional_on_servable')} for cohort in ('selected_probe','full_update_validation')}
            training[c['name']]=copy.deepcopy(baseline[c['name']])
            for cohort in training[c['name']]:
                for d in training[c['name']][cohort]:training[c['name']][cohort][d]={k:v-(.003 if c['method']=='historical' else .001) for k,v in metrics.items()}
        r=comparisons(baseline,training,p)
        self.assertAlmostEqual(r['paired']['full_update_validation']['full_denominator']['lr3e5_historical_minus_plain']['ndcg@10']['mean'],0)
        self.assertFalse(r['future_evaluated']);self.assertFalse(r['automatic_lr_selection'])
        training.pop(next(iter(training)))
        with self.assertRaises(ValueError):comparisons(baseline,training,p)


@unittest.skipUnless(HAVE,'Actual old/new LR GPT and complete no-Future acceptance required on server')
class RealModels(unittest.TestCase):
    def test_actual_four_arm_LR_ablation_private_copy_no_future_and_full_verifier(self):
        import torch
        from genrec.cf_letter_models import GenerativeRetriever
        from genrec.cf_letter_base_training import seed_all
        from genrec.cf_letter_downstream_protocol import load_training,save_arrays
        from genrec.cf_letter_downstream_training import train_arm
        from genrec.cf_letter_refresh_protocol import versioned_mapping
        from run_cf_letter_downstream import counters
        from run_cf_letter_multiseed import seal
        import verify_cf_letter_gpt_lr as verifier
        spec=importlib.util.spec_from_file_location('lr_real_fixture',ROOT/'tests/incremental/test_cf_letter_downstream.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        torch.set_num_threads(1);seed_all(2026)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);q,deployment,directory=fixture.temporal_fixture(root)
            p=plan(ROOT);p.update(seeds=[2027,2028],steps_per_arm=4,baseline_original_steps=4,optimizer_step_cap=16,selection_steps=[0,1,2,4],topk=[1,2,4],fixture_source_code='old-fixture')
            q.update(job_id=JOB,scope_id=SCOPE,steps_per_arm=4,optimizer_step_cap=16,selection_steps=[0,1,2,4],lr=3e-5,
                arms=[{k:v for k,v in c.items() if k not in ('steps','phase')} for c in cases(p)])
            (directory/'future_test.jsonl').write_text('MUST NOT OPEN');(directory/'anchor_holdout.jsonl').write_text('MUST NOT OPEN')
            train,val,probe,binding=load_training(deployment,q)
            source=root/'accepted';gpt=source/'gpt';(gpt/'inputs/base/gpt').mkdir(parents=True);(gpt/'mappings').mkdir()
            model=GenerativeRetriever(q['history'],q['quantizer']['capacity'],q['gpt'])
            torch.save(dict(model=model.state_dict(),step=2,code_sha='base-fixture'),gpt/'inputs/base/gpt/selected.pth')
            q['predecessor'].update(code_sha='base-fixture',gpt_step=2)
            count=q['old_catalog_items'];codes=np.column_stack([np.arange(count)%4,np.arange(count)//4%4,np.arange(count)//16%4]);old,_=allocate(codes,4);np.save(gpt/'inputs/base/item_sids.npy',old)
            for c in cases(p):
                fresh=codes.copy();fresh[:,0]=(fresh[:,0]+(1 if c['method']=='plain' else 2))%4;mapping,capacity=versioned_mapping(old,fresh,4)
                self.assertTrue(capacity['full_sid_allocation_feasible']);np.save(gpt/'mappings'/(c['name']+'.npy'),mapping)
                oldq=copy.deepcopy(q);oldq.update(lr=1e-4,seed=c['seed'])
                with patch.dict(os.environ,{'LAB_BUFFER_CODE_SHA':'old-fixture'}):train_arm(gpt,c,train,val,probe,oldq,'cpu',lambda:None)
            index={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()};bound={str(source/n):index[n] for n in source_files(p)}
            before={str(f):sha(f) for f in source.rglob('*') if f.is_file()}
            workspace=root/'workspaces'/JOB;output=workspace/'outputs/cf-gpt-lr';output.mkdir(parents=True)
            with patch('genrec.cf_letter_gpt_lr_protocol.check_source',return_value=(source,index,bound)),patch('genrec.cf_letter_gpt_lr_protocol.shutil.disk_usage',return_value=type('Disk',(),{'free':10**15})()):transfer(root,output,p)
            binding['input_sha256'].update(bound);save(output/'binding.json',binding)
            save_arrays(output/'prepared.npz',{**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()},'probe':probe})
            binding['prepared_sha256']=sha(output/'prepared.npz');save(output/'data_audit.json',binding)
            for c in cases(p):
                newq=copy.deepcopy(q);newq['seed']=c['seed']
                with patch.dict(os.environ,{'LAB_BUFFER_CODE_SHA':'new-fixture'}):train_arm(output,c,train,val,probe,newq,'cpu',lambda:None)
            save(output/'launch_journal.json',cases(p));publications(output,p,q)
            total,counts,intents=counters(output)
            report=dict(stage=SCOPE,job_id=JOB,code_sha='new-fixture',execution_status='completed',new_lr=3e-5,baseline_lr=1e-4,seeds=p['seeds'],
                original_inputs_unchanged=True,uncertain_step_records=[],stop_new_stages_reason=None,quantizer_training_performed=False,sasrec_training_performed=False,
                base_training_performed=False,holdout_evaluated=False,future_evaluated=False,optimizer_steps_spent=total,stage_step_counts=counts,
                local_artifact_index_sha256=seal(output),returned_artifact_sha256={n:sha(output/(n+'.json')) for n in ('data_audit','baseline','runs','curves','comparisons')})
            save(output/'compact.json',report);context=root/'context.json';save(context,dict(workspace=str(workspace),request={'code_sha':'new-fixture'},contract={'input_sha256':{str(deployment):sha(deployment)}},contract_sha256='fixture'))
            snapshot={str(f):sha(f) for f in output.rglob('*') if f.is_file()}
            with patch.object(verifier,'check_source',return_value=(source,index,bound)):
                verdict=verifier.verify(context,configuration=p,recipe=q)
            self.assertEqual(verdict['status'],'verified');self.assertEqual(verdict['actual_optimizer_steps'],16)
            self.assertTrue(all(sha(f)==h for f,h in before.items()));self.assertTrue(all(sha(f)==h for f,h in snapshot.items()))
            self.assertFalse(any(output.rglob('future_queries.jsonl')));self.assertIn('Recall@2',verdict['report_markdown'])


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Actual GPT LR/no-Future/complete-verifier preflight required')
    unittest.main()
