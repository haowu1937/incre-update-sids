"""Compact hard-gated training pilot results. Standard library only."""
import json
from pathlib import Path
import sys

def fmt(x):return format(x,'.9g') if isinstance(x,(int,float)) else str(x)

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python show_gate.py COMPARISON_DIR')
    r=json.loads((Path(sys.argv[1])/'summary.json').read_text())
    if r.get('stage')!='historical_hard_gate_training_comparison':raise ValueError('Unexpected comparison stage')
    w=r['work_counters']
    print('execution_status =',r['execution_status'],'| steps =',r['update_optimizer_steps'],'| rho =',r['rho'])
    print('same_base =',r['same_base'],'| encoder_unchanged_both =',r['encoder_unchanged_both'])
    print('same_content_optimizer_and_sampler =',r['same_content_optimizer_and_sampler'],'| same_compute =',r['same_compute'])
    print('full_anchor_train_queries =',r['queries'],'| holdout_used =',r['holdout_used'],'| ranking_evaluated =',r['ranking_evaluated'])
    print('historical_accepts =',w['committed_historical_steps'],'| content_fallbacks =',w['committed_content_steps'],'| identical =',w['identical_candidates'])
    print('local_guard_passed =',r['per_step_local_guard_passed'],'| independent_Plain_dominance_guaranteed =',r['independent_plain_endpoint_dominance_guaranteed'])
    print('historical_rejection_reasons =',w['rejected_historical_reasons'])
    print('arm positive_damage mean_delta_NLL RQ_MSE')
    for key in ('plain','ungated','hard_gated'):
        a=r.get(key)
        if a:
            h=a['history'];print(key,fmt(h['positive_damage']),fmt(h['mean_delta_sequence_nll']),fmt(a['rq_mse']))
    d=r['hard_gated_minus_plain']
    print('gated_minus_plain_positive_damage =',fmt(d['positive_damage']),'| training_cluster_CI95 =',d['positive_damage_cluster_ci95'])
    print('gated_minus_plain_mean_NLL =',fmt(d['mean_sequence_nll']),'| RQ_MSE_difference =',fmt(d['rq_mse']))
    c=r['hard_gated']['collision_statistics']
    print('gated_unique_codes =',c['unique_semantic_codes'],'| max_bucket =',c['maximum_collision_bucket'],'| CB_used =','/'.join(str(x['used']) for x in c['codebooks']))
    print('median_step_seconds =',fmt(r['median_step_seconds']))
    print('step content_candidate_H historical_candidate_H selected selected_H')
    rows=[json.loads(s) for s in (Path(r['rqvae_dir'])/'metrics.jsonl').read_text().splitlines() if s.strip()]
    for row in rows:
        if row['step']%10==0 or row['step']==r['update_optimizer_steps']:
            print(row['step'],fmt(row['content_candidate'].get('history',{}).get('positive_damage')),
                fmt(row['historical_candidate'].get('history',{}).get('positive_damage')),
                row['selected'],fmt(row['selected_history']['positive_damage']))
    print('Candidate content_H is local to the gated trajectory; it is not the independent Plain trajectory.')
    print('Training-only diagnostic; no held-out improvement claim.')

if __name__=='__main__':main()
