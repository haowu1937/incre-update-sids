"""Data/identity/capacity checks plus mandatory actual-model refresh preflight."""
import copy
import hashlib
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
from genrec.cf_letter_refresh_protocol import anchor_indices,damage,load_data,plan,submission,transfer_predecessor,versioned_mapping

HAVE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


class Protocol(unittest.TestCase):
    def test_scope_counts_and_generic_contract(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'scripts/lab_buffer'));import task_runtime
        p=plan(ROOT);self.assertEqual(p['optimizer_step_cap'],12240)
        self.assertEqual((104275+255)//256*30,12240)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'task-runtime-policy.json',dict(protocol_version=1,max_parallel=2,read_roots=[str(root)]))
            r=SimpleNamespace(root=root,config=dict(task_runtime_version=1,worker_id='server2'),read=lambda code,path:(ROOT/path).read_bytes() if (ROOT/path).is_file() else None)
            v=submission('a'*40,'b'*64,root,ROOT);task_runtime.validate(r,v['request'],encoded(v['request']),v['contract'],v['scope'])
            self.assertFalse(v['scope'].get('future_evaluation',False));self.assertEqual(len(v['contract']['preflight']),2)

    def test_same_mapping_preserves_every_suffix(self):
        codes=[[0,1,2],[0,1,2],[2,2,2]];previous,_=allocate(codes,4)
        current,stats=versioned_mapping(previous,codes,4)
        self.assertTrue(np.array_equal(current,previous));self.assertTrue(stats['full_sid_allocation_feasible'])

    def test_movers_append_after_reserved_old_slots(self):
        previous,_=allocate([[0,0,0],[0,0,0],[1,1,1]],4)
        current,stats=versioned_mapping(previous,[[1,1,1],[0,0,0],[1,1,1]],4)
        self.assertEqual(current[0,3],13);self.assertEqual(current[1,3],13);self.assertEqual(current[2,3],12)
        self.assertEqual(len(set(map(tuple,current))),3);self.assertTrue(stats['full_sid_allocation_feasible'])

    def test_reserved_capacity_can_overflow_without_large_current_bucket(self):
        previous,_=allocate([[0,0,0]]*4+[[1,1,1]]*2,4)
        current,stats=versioned_mapping(previous,[[2,2,2]]*4+[[0,0,0]]*2,4)
        self.assertIsNone(current);self.assertEqual(stats['maximum_current_collision_bucket'],4)
        self.assertEqual(stats['maximum_required_suffix_capacity'],6)
        self.assertEqual(json.loads(json.dumps(stats))['overflow_bucket_count'],1)
        with self.assertRaises(ValueError):versioned_mapping(previous,np.zeros((5,3),dtype=int),4)

    def test_damage_uses_positive_part_and_preserves_gains_in_delta(self):
        rows=[dict(user='a',old_sequence_nll=3.,new_sequence_nll=2.,history_changed=True),dict(user='b',old_sequence_nll=3.,new_sequence_nll=5.,history_changed=True)]
        a=damage(rows,50,2026);self.assertEqual(a['H'],1.);self.assertEqual(a['mean_delta_sequence_nll'],.5)
        same=[dict(row,new_sequence_nll=row['old_sequence_nll'],history_changed=False) for row in rows]
        self.assertEqual(damage(same,50,2026)['H'],0.)

    def test_dependency_copy_is_bounded_hash_verified_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=plan(ROOT);source=root/'workspaces'/p['predecessor']['job_id']/'outputs/cf-base';source.mkdir(parents=True)
            (source/'table.npy').write_bytes(b'accepted data');save(source/'compact.json',dict(research_outcome='base_chain_accepted',code_sha=p['predecessor']['code_sha']));save(source/'data_audit.json',{})
            p['predecessor'].update(files_sha256={'table.npy':sha(source/'table.npy')},compact_sha256=sha(source/'compact.json'),audit_sha256=sha(source/'data_audit.json'))
            result=transfer_predecessor(root,root/'new-output',p)
            self.assertEqual(sha(root/'new-output/inputs/base/table.npy'),sha(source/'table.npy'));self.assertEqual(len(result),3)
            (source/'table.npy').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):transfer_predecessor(root,root/'another-output',p)

    def test_cumulative_data_never_reads_validation_positives_or_future(self):
        from genrec.temporal import build_temporal_protocol
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);project=root/'project';directory=project/'cache/incremental/Beauty/beauty_20261007T104452Z';directory.mkdir(parents=True)
            protocol=build_temporal_protocol([('u'+str(i%8),'item'+str(i%53),1000000000+i,None) for i in range(800)])
            save(directory/'protocol_manifest.json',protocol['manifest']);save(directory/'registries.json',protocol['registries'])
            for split in ('base_train','update_train','update_val'):
                (directory/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in protocol['examples'][split]))
            (directory/'future_test.jsonl').write_text('MUST NOT READ');(directory/'base_val.jsonl').write_text('MUST NOT READ')
            names=['protocol_manifest.json','registries.json','base_train.jsonl','update_train.jsonl','update_val.jsonl']
            save(directory/'summary.json',dict(execution_status='completed',protocol_sha256=protocol['manifest']['protocol_sha256'],artifact_files_sha256={n:sha(directory/n) for n in names}))
            p=plan(ROOT);p['old_catalog_items']=len(protocol['registries']['base_train']['id2item'])-1;p['expected_queries']={s:len(protocol['examples'][s]) for s in p['allowed_query_splits']};p['anchor_size']=8
            e1=project/'e1';e1.mkdir();rows=protocol['examples']['base_train'];indices=anchor_indices(rows,8,2026)
            (e1/'anchor_train.jsonl').write_text(''.join(json.dumps(dict(source_row=int(i),**rows[i]))+'\n' for i in indices))
            paths=[directory/n for n in names+['summary.json']]+[e1/'anchor_train.jsonl'];deployment=root/'main-deployment.json'
            save(deployment,dict(project=str(project),protocol_dir=str(directory),e1_dir=str(e1),input_sha256={str(f):sha(f) for f in paths}))
            data,binding=load_data(deployment,p)
            self.assertEqual(len(data['train']['targets']),len(protocol['examples']['base_train'])+len(protocol['examples']['update_train']))
            self.assertEqual(binding['items'][:p['old_catalog_items']],protocol['registries']['base_train']['id2item'][1:])
            self.assertTrue(np.all(data['val']['targets']<=p['old_catalog_items']))
            self.assertTrue(np.all(data['train']['targets']>0));self.assertEqual(len(data['anchors']['targets']),8)
            self.assertFalse(any('future_test.jsonl' in name for name in binding['input_sha256']))


@unittest.skipUnless(HAVE,'Torch/transformers unavailable; mandatory actual-model tests run on server')
class RealModels(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        from genrec.cf_letter_base_training import seed_all
        seed_all(2026)

    def test_exact_old_rows_and_hidden_weights_survive_expansion(self):
        import torch
        from genrec.cf_letter_models import SASRec
        from genrec.cf_letter_refresh_training import expanded_sasrec
        p=plan(ROOT);p['history']=3;p['sasrec'].update(dim=8,layers=1,heads=2,ffn=16)
        old=SASRec(32,3,p['sasrec']);new=expanded_sasrec({'model':old.state_dict()},36,p,'cpu')
        self.assertTrue(torch.equal(old.items.weight,new.items.weight[:33]));self.assertEqual(new.items.num_embeddings,37)
        for name,value in old.state_dict().items():
            if name!='items.weight':self.assertTrue(torch.equal(value,new.state_dict()[name]))
        self.assertFalse(torch.equal(new.items.weight[33:],torch.zeros_like(new.items.weight[33:])))

    def test_actual_refresh_baseline_and_feature_only_are_frozen_and_reloadable(self):
        import torch
        from genrec.cf_letter_models import SASRec,GenerativeRetriever,CollaborativeRQ
        from genrec.cf_letter_refresh_training import baseline,refresh,feature_only
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);base=output/'inputs/base';base.mkdir(parents=True);p=plan(ROOT)
            p.update(history=3,batch_size=16,eval_batch_size=4,probe_size=8,anchor_size=8,optimizer_step_cap=6,bootstrap_replicates=10,old_catalog_items=32)
            p['sasrec'].update(dim=8,layers=1,heads=2,ffn=16,epochs=2,step_cap=6,eval_every_epochs=1,dropout=.1)
            p['gpt'].update(d_model=16,layers=1,heads=2,ffn=32,user_buckets=8,dropout=.1);p['quantizer'].update(dim=8,capacity=32)
            p['predecessor'].update(code_sha='test',sasrec_step=2,gpt_step=2)
            old=SASRec(32,3,p['sasrec']);features=old.items.weight[1:].detach().numpy().copy();np.save(base/'cf_embeddings.npy',features)
            q=CollaborativeRQ(torch.tensor(features),p['quantizer']);q.initialize();codes=q.assignments().numpy();mapping,_=allocate(codes,32)
            np.save(base/'item_sids.npy',mapping);(base/'quantizer').mkdir();np.save(base/'quantizer/semantic_codes.npy',codes)
            for name,model in [('sasrec',old),('gpt',GenerativeRetriever(3,32,p['gpt'])),('quantizer',q)]:
                (base/name).mkdir(exist_ok=True);torch.save(dict(model=model.state_dict(),step=2,code_sha='test'),base/name/'selected.pth')
            hashes={str(path):sha(path) for path in base.rglob('*') if path.is_file()}
            def data(n):return dict(history=np.asarray([[0,(i%32)+1,((i+1)%32)+1] for i in range(n)]),targets=np.asarray([((i+2)%32)+1 for i in range(n)]),users=np.arange(n)%8,samples=np.asarray(['sample'+str(i) for i in range(n)]))
            values=dict(train=data(48),val=data(8),anchors=data(8),probe=np.arange(8),anchor_users=np.asarray(['u'+str(i%4) for i in range(8)]),anchor_source_rows=np.arange(8))
            values['train']['targets'][-4:]=np.arange(33,37);values['val']['targets'][-1]=0
            binding=dict(items=['item'+str(i) for i in range(36)],old_items=['item'+str(i) for i in range(32)])
            b=baseline(output,values,p,binding,'cpu');self.assertEqual(b['anchor_train']['H'],0.)
            s=refresh(output,values,p,binding,'cpu');self.assertEqual(s['optimizer_steps'],6);self.assertEqual(s['epochs'],2)
            f=feature_only(output,values,p,binding,'cpu');self.assertTrue(f['teacher_unchanged']);self.assertTrue(f['codebooks_unchanged'])
            self.assertFalse(f['new_item_sids_allocated']);self.assertEqual(hashes,{name:sha(name) for name in hashes})
            self.assertEqual(sum(read(output/'sasrec/negative_exposure.json')['negative_counts']),96)


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Actual Torch/GPT checks are mandatory; mocks/skips do not authorize Beauty training')
    unittest.main()
