"""Pure data/identity layer for the bounded CF-LETTER Base acceptance task."""
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

MANIFEST = 'docs/experiments/cf_letter_base_acceptance.json'
SCOPE = 'cf-letter-base-v1'
JOB = 'cf-letter-base-001'


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def submission(code_sha,deployment_sha,root,repository):
    """Pure constructor for the already installed generic v1 task interface."""
    root=Path(root)
    request=dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code_sha,
        created_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=86400,
        argv=['python','-u','-B','run_cf_letter_base.py','--deployment',str(root/'main-deployment.json'),'--output','outputs/cf-base'],
        artifacts=['outputs/task-runtime.json','outputs/cf-base/compact.json','outputs/cf-base/data_audit.json','outputs/cf-base/curves.json'])
    scope=dict(protocol_version=1,scope_id=SCOPE,max_optimizer_steps=54640,max_parallel=1,
        authorization='Hao: first accept only the Base chain; SASRec <=150 epochs, CF-RQ <=1000 steps, Base GPT <=30 epochs; no Update/Historical/Future.',
        allowed_splits=['base_train','base_val'],stop_after_base=True)
    contract=dict(protocol_version=1,scope_id=SCOPE,job_id=JOB,code_sha=code_sha,
        request_sha256=hashlib.sha256(encoded(request)).hexdigest(),scope_path='task-scopes/'+SCOPE+'.json',
        scope_sha256=hashlib.sha256(encoded(scope)).hexdigest(),manifest={'path':MANIFEST,'sha256':sha(Path(repository)/MANIFEST)},
        preflight=[['python','-u','-B','tests/incremental/test_cf_letter_base.py','--require-torch']],
        verifier='verify_cf_letter_base.py',optimizer_step_reservation=54640,
        input_sha256={str(root/'main-deployment.json'):deployment_sha})
    return dict(request=request,scope=scope,contract=contract)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.tmp')
    with pending.open('w') as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)


def plan(root):
    p = read(Path(root)/MANIFEST)
    if (p['scope_id'] != SCOPE or p['job_id'] != JOB or p['allowed_splits'] != ['base_train','base_val']
            or p['optimizer_step_cap'] != 54640 or p['history'] != 20 or p['seed'] != 2026
            or any(p[k] is not False for k in ('update_training','historical_training','future_evaluation','retry'))):
        raise ValueError('Unsupported Base-only scope')
    if p['sasrec']['step_cap']+p['quantizer']['steps']+p['gpt']['step_cap'] != p['optimizer_step_cap']:
        raise ValueError('Stage budget mismatch')
    return p


def safe_file(directory, name):
    p = Path(directory)/name
    if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError('Unsafe input file '+name)
    return p


def encode_rows(rows, registry, history, user_buckets):
    """Fixed-width, right-aligned raw events; masks never compact unknown events."""
    n = len(rows)
    ids = np.zeros((n, history), dtype=np.int64)
    targets = np.zeros(n, dtype=np.int64)
    users = np.zeros(n, dtype=np.int64)
    samples = []; seen = set()
    for i, row in enumerate(rows):
        sample = row['sample_id']; seq = row['item_seq']; times = row['time_seq']
        if (sample in seen or len(seq) < 2 or len(seq) != len(times)
                or row['target_time'] != times[-1] or any(t >= times[-1] for t in times[:-1])):
            raise ValueError('Noncausal/duplicate explicit query')
        seen.add(sample); samples.append(sample)
        values = [registry.get(x, 0) for x in seq[:-1][-history:]]
        ids[i, -len(values):] = values
        targets[i] = registry.get(seq[-1], 0)
        users[i] = int(hashlib.sha256(row['user'].encode()).hexdigest(),16) % user_buckets
    return dict(history=ids, targets=targets, users=users, samples=np.asarray(samples))


def load_base(deployment, p):
    """Verify transitive bindings; never open Update/Future query files."""
    d = read(deployment); directory = Path(d['protocol_dir']).resolve()
    expected_dir = Path(d['project'])/'cache/incremental/Beauty/beauty_20261007T104452Z'
    if directory != expected_dir.resolve():
        raise ValueError('Unexpected temporal input directory')
    binding = {}
    for name in ('summary.json','protocol_manifest.json','registries.json','base_train.jsonl','base_val.jsonl'):
        path = safe_file(directory, name); actual = sha(path)
        if d['input_sha256'].get(str(path)) != actual:
            raise ValueError('Deployment-bound temporal input changed: '+name)
        binding[str(path)] = actual
    summary = read(directory/'summary.json'); manifest = read(directory/'protocol_manifest.json')
    for name in ('protocol_manifest.json','registries.json','base_train.jsonl','base_val.jsonl'):
        if summary['artifact_files_sha256'][name] != binding[str(directory/name)]:
            raise ValueError('Temporal preparation checksum mismatch')
    if summary['execution_status'] != 'completed' or summary['protocol_sha256'] != manifest['protocol_sha256']:
        raise ValueError('Incomplete temporal protocol')
    registry = read(directory/'registries.json')['base_train']
    raw = registry['id2item'][1:]; mapping = registry['item2id']
    if len(raw) != p['expected_catalog_items'] or any(mapping[x] != i+1 for i,x in enumerate(raw)):
        raise ValueError('Base catalog row order mismatch')
    data = {}; all_ids = set()
    for split in ('base_train','base_val'):
        with (directory/(split+'.jsonl')).open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != p['expected_queries'][split]:
            raise ValueError('Base split count mismatch')
        cut = manifest['cutoffs_exclusive']['base_train' if split=='base_train' else 'base']
        lower = manifest['windows'][split]['first_timestamp']
        if any(not lower <= r['target_time'] < cut for r in rows):
            raise ValueError('Target outside its authorized Base window')
        data[split] = encode_rows(rows, mapping, p['history'], p['gpt']['user_buckets'])
        if all_ids.intersection(data[split]['samples']):raise ValueError('Split query overlap')
        all_ids.update(data[split]['samples'])
    if np.any(data['base_train']['targets']==0) or np.any(data['base_train']['history'][:,-1]==0):
        raise ValueError('Unknown Base training item')
    probe = sorted(sorted(range(len(data['base_val']['targets'])), key=lambda i: hashlib.sha256(
        (str(p['seed'])+':'+data['base_val']['samples'][i]).encode()).hexdigest())[:p['probe_size']])
    data['probe'] = np.asarray(probe, dtype=np.int64)
    return d, data, dict(input_sha256=binding, protocol_sha256=manifest['protocol_sha256'],
        items=raw, probe_sample_ids=data['base_val']['samples'][probe].tolist(),
        allowed_splits=['base_train','base_val'], history=p['history'])


def allocate(codes, capacity):
    counts = {}; result = []
    for row in codes:
        prefix = tuple(int(v) for v in row)
        if len(prefix)!=3 or any(not 0 <= v < capacity for v in prefix):raise ValueError('Invalid semantic code')
        suffix = counts.get(prefix,0)
        if suffix >= capacity:raise ValueError('Base collision suffix capacity exhausted')
        result.append([*(v+i*capacity for i,v in enumerate(prefix)),3*capacity+suffix])
        counts[prefix] = suffix+1
    return np.asarray(result,dtype=np.int64), dict(items=len(result), unique_semantic_codes=len(counts),
        maximum_collision_bucket=max(counts.values()), suffix_capacity=capacity, unique_full_sids=len(result))


def gpt_arrays(data, mapping, capacity, user_buckets):
    # Item-row 0 is unknown/PAD; SID token 0 is a valid first codeword.
    vocab = 4*capacity+user_buckets+10; pad = vocab-1
    lookup = np.full((len(mapping)+1,4),pad,dtype=np.int64); lookup[1:] = mapping
    b,h = data['history'].shape
    prefix = np.concatenate([(4*capacity+data['users'])[:,None],lookup[data['history']].reshape(b,4*h)],axis=1)
    mask = np.concatenate([np.ones((b,1),dtype=bool),np.repeat(data['history']!=0,4,axis=1)],axis=1)
    return prefix,mask,lookup[data['targets']]


def metrics(records, topk):
    total = len(records); served = sum(r['target']>0 for r in records)
    if not total:raise ValueError('Empty ranking query set')
    values = {f'{name}@{k}':0. for name in ('recall','ndcg') for k in topk}
    nll = 0.
    for row in records:
        if row['target']<=0:
            if row['predictions'] or row['nll'] is not None:raise ValueError('Cold target must be an explicit miss')
            continue
        preds = row['predictions']
        if len(preds)!=max(topk) or len(set(preds))!=len(preds) or not math.isfinite(row['nll']):raise ValueError('Invalid ranking record')
        nll += row['nll']
        rank = preds.index(row['target'])+1 if row['target'] in preds else None
        for k in topk:
            if rank is not None and rank<=k:
                values[f'recall@{k}'] += 1
                values[f'ndcg@{k}'] += 1/math.log2(rank+1)
    return dict(all_queries=total,servable_queries=served,unavailable_targets=total-served,
        full_denominator={k:v/total for k,v in values.items()},
        conditional_on_servable={k:v/served if served else None for k,v in values.items()},
        mean_sequence_nll_servable=nll/served if served else None,invalid_predictions=0,
        ranking_catalog='base_train',seen_item_filter=False)
