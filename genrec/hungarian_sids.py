"""Post-tokenization, position-preserving bijections; no model/data fitting."""
import hashlib
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate(mapping,capacities):
    if not capacities or any(type(k) is not int or k<1 for k in capacities):raise ValueError('Invalid codebook capacities')
    if not mapping or len(set(map(tuple,mapping.values())))!=len(mapping):raise ValueError('Full SIDs must be unique')
    offsets=np.cumsum([1,*capacities[:-1]])
    for tokens in mapping.values():
        if len(tokens)!=len(capacities) or any(type(v) is not int or not a<=v<a+k for v,a,k in zip(tokens,offsets,capacities)):
            raise ValueError('Wrong SID position/vocabulary')


def fit(old,new,capacities):
    """Maximize matched-item token overlap, then fixed token labels among ties.

    Fit only supplied shared old-catalog identities. All K tokens are covered,
    including inactive tokens; the sum of secondary diagonal terms cannot
    outweigh one primary match. No distance, model score or Future is used.
    """
    validate(old,capacities);validate(new,capacities)
    common=sorted(set(old)&set(new))
    if not common:raise ValueError('No common historical items')
    permutations=[];books=[];offset=1
    for level,k in enumerate(capacities):
        weights=np.zeros((k,k),dtype=np.int64)
        for item in common:weights[new[item][level]-offset,old[item][level]-offset]+=1
        rows,cols=linear_sum_assignment(weights*(k+1)+np.eye(k,dtype=np.int64),maximize=True)
        permutation=np.empty(k,dtype=np.int64);permutation[rows]=cols
        permutations.append(permutation.tolist())
        books.append(dict(position=level,capacity=k,before_matching_tokens=int(np.trace(weights)),
            after_matching_tokens=int(weights[rows,cols].sum()),active_new=int((weights.sum(1)>0).sum()),active_old=int((weights.sum(0)>0).sum())))
        offset+=k
    return dict(format='positionwise_hungarian_full_sid_v1',capacities=capacities,permutations=permutations,
        common_items=len(common),common_items_sha256=canonical(common),objective='maximum unweighted shared-item token overlap; identity secondary tie-break',
        positions=books,suffix_policy='fourth position is globally bijectively relabeled; never reallocate suffix after translation')


def transform(mapping,translation,inverse=False):
    capacities=translation['capacities'];validate(mapping,capacities);offset=1;permutations=[]
    if len(translation['permutations'])!=len(capacities):raise ValueError('Missing position mapping')
    for values,k in zip(translation['permutations'],capacities):
        if sorted(values)!=list(range(k)):raise ValueError('Not a full per-position bijection')
        a=np.asarray(values);permutations.append(np.argsort(a) if inverse else a)
    out={}
    for item,tokens in mapping.items():
        offset=1;row=[]
        for token,k,p in zip(tokens,capacities,permutations):row.append(int(p[token-offset])+offset);offset+=k
        out[item]=row
    validate(out,capacities)
    if sorted(bucket_sizes(mapping))!=sorted(bucket_sizes(out)):raise ValueError('Translation altered semantic collisions')
    return out


def bucket_sizes(mapping):
    from collections import Counter
    return list(Counter(tuple(v[:3]) for v in mapping.values()).values())
