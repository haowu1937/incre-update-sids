"""Causality/lock/selection/paired-metric checks and actual five-arm CPU acceptance."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
import numpy as np
from genrec.cf_letter_protocol import save,read,sha,encoded,allocate
from genrec.cf_letter_downstream_protocol import JOB,SCOPE,MANIFEST,plan,cases,submission,load_training,load_stage,valid_lock,select,aggregate,paired,transfer
HAVE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


def temporal_fixture(root):
    from genrec.temporal import build_temporal_protocol
    project=root/'project';directory=project/'cache/incremental/Beauty/beauty_20261007T104452Z';directory.mkdir(parents=True)
    reviews=[('u'+str(i%8),'item'+str(i%(16 if i<600 else 24 if i<800 else 32)),1000000000+i,None) for i in range(1000)]
    protocol=build_temporal_protocol(reviews);save(directory/'protocol_manifest.json',protocol['manifest']);save(directory/'registries.json',protocol['registries'])
    for stage,rows in protocol['examples'].items():(directory/(stage+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    names=['protocol_manifest.json','registries.json',*[s+'.jsonl' for s in protocol['examples']]]
    save(directory/'summary.json',dict(execution_status='completed',protocol_sha256=protocol['manifest']['protocol_sha256'],artifact_files_sha256={n:sha(directory/n) for n in names}))
    deployment=root/'main-deployment.json';save(deployment,dict(project=str(project),protocol_dir=str(directory),runner_root=str(root),input_sha256={str(directory/n):sha(directory/n) for n in [*names,'summary.json']}))
    p=plan(ROOT);p.pop('zero_step_repair_of',None);p.update(old_catalog_items=len(protocol['registries']['base_train']['id2item'])-1,
        expected_queries={s:len(protocol['examples'][s]) for s in ('update_train','update_val','future_test')},history=2,probe_size=4,
        batch_size=4,eval_batch_size=2,steps_per_arm=4,optimizer_step_cap=20,selection_steps=[0,1,2,4],selection_metric='ndcg@2',topk=[1,2,4],beam=4,bootstrap_replicates=10)
    p['gpt'].update(d_model=16,layers=1,heads=2,ffn=32,user_buckets=8);p['quantizer'].update(dim=8,capacity=4)
    return p,deployment,directory


def lock_fixture(p):return dict(job_id=p['job_id'],future_used_for_selection=False,arms={v['name']:dict(selected_step=0,selected_sha256='a'*64,mapping_sha256='b'*64) for v in p['arms']})


class Protocol(unittest.TestCase):
    def test_report_uses_configured_cutoffs_for_tiny_and_production(self):
        from genrec.cf_letter_downstream_protocol import future_table
        for cutoffs in ([1,2,4],[5,10,20]):
            p=plan(ROOT);p['topk']=cutoffs
            training={v['name']:{'optimizer_steps':4} for v in p['arms']}
            future={v['name']:dict(selected_update_step=2,coverage=.5,full_denominator={f'{m}@{k}':.1 for m in ('recall','ndcg') for k in cutoffs}) for v in p['arms']}
            table=future_table(training,future,p);self.assertEqual(len(table),7)
            self.assertIn('Recall@'+str(cutoffs[0]),table[0]);self.assertIn('NDCG@'+str(cutoffs[-1]),table[0])
            self.assertTrue(all(len(row.split('|'))==len(table[0].split('|')) for row in table))

    def test_repair_requires_exact_zero_real_update_evidence(self):
        from genrec.cf_letter_downstream_protocol import verify_zero_step_failure
        p=plan(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proof=p['zero_step_repair_of'];path=root/'workspaces'/proof['job_id']/'outputs/task-runtime.json'
            runtime=dict(job_id=proof['job_id'],code_sha=proof['code_sha'],status='failed',checks_completed=1,command_started=False)
            save(path,runtime);proof['runtime_sha256']=sha(path);verify_zero_step_failure(root,p)
            runtime['command_started']=True;save(path,runtime);proof['runtime_sha256']=sha(path)
            with self.assertRaises(ValueError):verify_zero_step_failure(root,p)
            runtime['command_started']=False;save(path,runtime);proof['runtime_sha256']=sha(path);(path.parent/'cf-downstream').mkdir()
            with self.assertRaises(ValueError):verify_zero_step_failure(root,p)

    def test_new_entry_scheduler_enforces_two_children_and_counts_single_level_steps(self):
        import time
        import run_cf_letter_optimizer as scheduler_module
        from run_cf_letter_downstream import counters,publish_training
        p=plan(ROOT);live=set();launched=[];peak=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'out';(output/'inputs').mkdir(parents=True);(output/'inputs/source').write_bytes(b'private source')
            (output/'prepared.npz').write_bytes(b'fixture');save(output/'binding.json',{});save(root/'deployment.json',dict(runner_root=str(root)))
            class Process:
                def __init__(self,argv,**kwargs):
                    self.pid=300+len(launched);self.name=argv[-1];self.returncode=None;self.polls=0
                    test.assertEqual(Path(argv[3]).name,'run_cf_letter_downstream.py');launched.append(self.name);live.add(self.pid);peak.append(len(live))
                def poll(self):
                    self.polls+=1
                    if self.polls<3:return None
                    if self.returncode is None:
                        live.remove(self.pid);self.returncode=0;folder=output/'arms'/self.name
                        save(folder/'optimizer_steps.json',dict(optimizer_steps_completed=2000,optimizer_step_intent=None))
                        summary=dict(case=next(v for v in cases(p) if v['name']==self.name));save(folder/'arm_result.json',dict(result=summary));save(folder/(self.name+'_curves.json'),summary)
                    return 0
            test=self
            with patch.object(scheduler_module,'gpu',side_effect=lambda:(set(live),6400)),patch.object(scheduler_module,'paused',return_value=False),patch.object(scheduler_module.subprocess,'Popen',Process),patch.object(scheduler_module.time,'sleep',return_value=None):
                scheduler=scheduler_module.Scheduler(output,root/'deployment.json',p,{'max_gpu_memory_mib':0},time.monotonic(),job_id=JOB,child_script='run_cf_letter_downstream.py',publisher=publish_training,counter=counters)
                self.assertEqual(len(scheduler.run(cases(p))),5)
            self.assertEqual(max(peak),2);self.assertFalse(live);self.assertEqual(counters(output)[0],10000)
            self.assertEqual(list(read(output/'training.json')),[v['name'] for v in p['arms']])

    def test_exact_five_arm_budget_and_existing_generic_contract(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'scripts/lab_buffer'));import task_runtime
        p=plan(ROOT);self.assertEqual(sum(c['steps'] for c in cases(p)),10000)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'task-runtime-policy.json',dict(protocol_version=1,max_parallel=2,read_roots=[str(root)]))
            runner=SimpleNamespace(root=root,config=dict(task_runtime_version=1,worker_id='server2'),read=lambda code,path:(ROOT/path).read_bytes() if (ROOT/path).is_file() else None)
            b=submission('a'*40,'b'*64,root,ROOT);task_runtime.validate(runner,b['request'],encoded(b['request']),b['contract'],b['scope'])
            self.assertEqual(len(b['request']['artifacts']),7)
            for key,value in [('steps_per_arm',2001),('optimizer_step_cap',10001),('quantizer_training',True),('selection_steps',[0,2000]),('parallel',3)]:
                q=copy.deepcopy(p);q[key]=value;save(root/MANIFEST,q)
                with self.assertRaises(ValueError):plan(root)

    def test_training_does_not_open_future_and_uses_same_old_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,deployment,directory=temporal_fixture(Path(tmp))
            # Poison both Future and holdout bytes: fitting must not touch them.
            (directory/'future_test.jsonl').write_text('DO NOT OPEN');(directory/'anchor_holdout.jsonl').write_text('DO NOT OPEN')
            train,val,probe,audit=load_training(deployment,p)
            self.assertTrue(np.all(train['targets']>0));self.assertTrue(np.all(train['targets']<=p['old_catalog_items']))
            self.assertGreater(audit['excluded_update_train_queries'],0)
            self.assertEqual(len(probe),4);self.assertEqual(len(val['targets']),p['expected_queries']['update_val'])
            self.assertFalse(any('future' in name for name in audit['input_sha256']))
            with self.assertRaisesRegex(ValueError,'locked'):load_stage(deployment,p,'future_test')
            with self.assertRaisesRegex(ValueError,'checksum'):load_stage(deployment,p,'future_test',lock_fixture(p))

    def test_global_lock_all_arms_and_earliest_tie_are_mandatory(self):
        p=plan(ROOT);lock=lock_fixture(p);valid_lock(lock,p)
        lock['arms'].pop('historical_b0')
        with self.assertRaises(ValueError):valid_lock(lock,p)
        curve=[dict(step=s,metrics={'full_denominator':{'ndcg@10':.1 if s in (100,500) else .05}}) for s in p['selection_steps']]
        self.assertEqual(select(curve,p)['step'],100)
        with self.assertRaises(ValueError):select(curve[:-1],p)

    def test_cold_target_full_denominator_and_user_cluster_pairing(self):
        p=plan(ROOT);p.update(topk=[1,2],beam=2,old_catalog_items=4,bootstrap_replicates=20)
        old=[dict(sample_id='a',user='u',target=2,predictions=[1,2],nll=2.,all_history_unavailable=False),dict(sample_id='b',user='v',target=0,predictions=[],nll=None,all_history_unavailable=True)]
        new=copy.deepcopy(old);new[0]['predictions']=[2,1]
        result=aggregate(new,p);self.assertEqual(result['full_denominator']['recall@1'],.5);self.assertEqual(result['coverage'],.5)
        difference=paired(old,new,p);self.assertEqual(difference['metrics']['recall@1']['absolute_difference'],.5)
        bad=copy.deepcopy(new);bad[0]['sample_id']='other'
        with self.assertRaises(ValueError):paired(old,bad,p)
        bad=copy.deepcopy(new);bad[0]['predictions']=[2,99]
        with self.assertRaises(ValueError):aggregate(bad,p)


@unittest.skipUnless(HAVE,'Torch/transformers unavailable; actual five-arm verifier preflight mandatory on server')
class RealModels(unittest.TestCase):
    def test_actual_five_arm_training_locked_future_and_full_scientific_acceptance(self):
        import time,shutil,torch
        from genrec.cf_letter_models import GenerativeRetriever
        from genrec.cf_letter_base_training import seed_all
        from genrec.cf_letter_refresh_protocol import versioned_mapping
        from genrec.cf_letter_downstream_protocol import save_arrays
        from genrec.cf_letter_downstream_training import train_arm,future_arm
        from run_cf_letter_downstream import publish_training,counters
        import verify_cf_letter_downstream as verifier
        torch.set_num_threads(1);seed_all(2026)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,deployment,directory=temporal_fixture(root)
            old_n=p['old_catalog_items'];codes=np.column_stack([np.arange(old_n)%4,np.arange(old_n)//4%4,np.arange(old_n)//16%4]);old_map,_=allocate(codes,4)
            base=root/'workspaces'/p['predecessor']['job_id']/'outputs/cf-base';(base/'gpt').mkdir(parents=True)
            model=GenerativeRetriever(p['history'],4,p['gpt']);torch.save(dict(model=model.state_dict(),step=2,code_sha='base-fixture'),base/'gpt/selected.pth')
            np.save(base/'item_sids.npy',old_map);save(base/'compact.json',dict(code_sha='base-fixture',stages={'gpt':{'selected_step':2}}))
            p['predecessor'].update(code_sha='base-fixture',gpt_step=2,compact_sha256=sha(base/'compact.json'),files_sha256={n:sha(base/n) for n in ('gpt/selected.pth','item_sids.npy')})
            source=root/'workspaces'/p['pair_predecessor']['job_id']/'outputs/cf-optimizer-pair';source.mkdir(parents=True);index={}
            for arm,shift in zip(p['arms'][1:],[(1,0),(1,1),(2,0),(2,1)]):
                folder=source/'arms'/arm['source']/arm['source'];folder.mkdir(parents=True);raw=codes.copy();raw[:,0]=(raw[:,0]+shift[0])%4;raw[:,1]=(raw[:,1]+shift[1])%4
                mapping,capacity=versioned_mapping(old_map,raw,4);self.assertTrue(capacity['full_sid_allocation_feasible'])
                np.save(folder/'item_sids_1000.npy',mapping);np.save(folder/'semantic_codes_1000.npy',raw);torch.save({'step':1000},folder/'last.pth')
                for f in folder.iterdir():index[str(f.relative_to(source))]=sha(f)
            save(source/'local_artifact_index.json',index);save(source/'compact.json',dict(code_sha='pair-fixture',execution_status='completed',optimizer_steps_spent=4000,uncertain_step_records=[]))
            p['pair_predecessor'].update(code_sha='pair-fixture',compact_sha256=sha(source/'compact.json'),index_sha256=sha(source/'local_artifact_index.json'))
            output=root/'outputs/cf-downstream';output.mkdir(parents=True);bound=transfer(root,output,p)
            train,val,probe,binding=load_training(deployment,p);bound.update(binding['input_sha256']);binding['input_sha256']=bound
            arrays={**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()},'probe':probe};save_arrays(output/'prepared.npz',arrays)
            training={}
            for case in cases(p):
                training[case['name']]=train_arm(output,case,train,val,probe,p,'cpu',lambda:None)
                shutil.copyfile(output/'arms'/case['name']/(case['name']+'_curves.json'),output/(case['name']+'_curves.json'))
            publish_training(output,p);save(output/'launch_journal.json',cases(p))
            lock=dict(job_id=JOB,code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'),created_at=time.time(),future_used_for_selection=False,arms={n:dict(selected_step=v['selected_update_step'],selected_sha256=v['selected_sha256'],mapping_sha256=v['mapping_sha256']) for n,v in training.items()})
            save(output/'selection_locked.json',lock);lock_sha=sha(output/'selection_locked.json');future,other=load_stage(deployment,p,'future_test',lock);bound.update(other)
            save_arrays(output/'future_prepared.npz',future);save(output/'future_opened.json',dict(time=time.time(),selection_lock_sha256=lock_sha));save(output/'data_audit.json',binding)
            futures={case['name']:future_arm(output,case,future,p,'cpu',lock_sha,lambda:None) for case in cases(p)};save(output/'future.json',futures)
            rows={name:[json.loads(line) for line in (output/'arms'/name/'future_queries.jsonl').read_text().splitlines()] for name in futures}
            comp={after+'_minus_'+before:paired(rows[before],rows[after],p) for before,after in [('plain_b09','historical_b09'),('plain_b0','historical_b0'),*[('ft_old',n) for n in futures if n!='ft_old']]};save(output/'comparisons.json',comp)
            index={str(f.relative_to(output)):sha(f) for f in output.rglob('*') if f.is_file()};save(output/'local_artifact_index.json',index)
            total,steps,intents=counters(output)
            report=dict(stage=SCOPE,job_id=JOB,code_sha=os.environ.get('LAB_BUFFER_CODE_SHA'),execution_status='completed',original_inputs_unchanged=True,
                selection_locked_before_future=True,future_used_for_selection=False,quantizer_training_performed=False,sasrec_training_performed=False,holdout_evaluated=False,
                returned_artifact_sha256={},local_artifact_index_sha256=sha(output/'local_artifact_index.json'),selection_lock_sha256=lock_sha,
                optimizer_steps_spent=total,stage_step_counts=steps,uncertain_step_records=intents)
            save(output/'compact.json',report);context=root/'context.json';save(context,dict(workspace=str(root),request={'code_sha':report['code_sha']},contract={'input_sha256':{str(deployment):sha(deployment)}},contract_sha256='fixture'))
            before={str(f):sha(f) for f in output.rglob('*') if f.is_file()}
            with patch.object(verifier,'plan',return_value=p):verdict=verifier.verify(context)
            self.assertEqual(verdict['status'],'verified');self.assertEqual(verdict['actual_optimizer_steps'],20)
            self.assertTrue(all(sha(f)==value for f,value in before.items()))
            for name in futures:self.assertEqual(futures[name]['all_queries'],p['expected_queries']['future_test'])


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Actual five-arm GPT training/beam/Future/verifier checks required before Beauty')
    unittest.main()
