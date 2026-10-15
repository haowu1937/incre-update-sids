"""Print compact matched frozen-encoder pilot results (standard library only)."""
import json
from pathlib import Path
import sys

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python show_frozen.py COMPARISON_DIR')
    directory=Path(sys.argv[1]);r=json.loads((directory/'summary.json').read_text())
    if r.get('stage')!='frozen_encoder_matched_training_pilot':raise ValueError('Unexpected comparison report')
    print('execution_status =',r['execution_status'],'| steps =',r['update_optimizer_steps'],'| rho =',r['rho'])
    print('same_base =',r['same_starting_base_checkpoint'],'| encoder_unchanged_both =',r['encoder_frozen_and_unchanged_both'])
    print('same_content_optimizer_and_sampler =',r['same_content_optimizer_and_sampler'],'| same_compute =',r['same_compute'])
    print('anchor_train_queries =',r['queries'],'| holdout_used =',r['holdout_used'],'| ranking_evaluated =',r['ranking_evaluated'])
    print('arm positive_damage mean_delta_NLL RQ_MSE unique_codes max_bucket CB_used')
    for name in ('plain','history_aware'):
        a=r[name];h=a['history'];c=a['collision_statistics']
        print(name,format(h['positive_damage'],'.9g'),format(h['mean_delta_sequence_nll'],'.9g'),
            format(a['rq_mse'],'.10g'),c['unique_semantic_codes'],c['maximum_collision_bucket'],
            '/'.join(str(x['used']) for x in c['codebooks']))
    d=r['history_aware_minus_plain']
    print('history_aware_minus_plain_positive_damage =',format(d['positive_damage'],'.9g'))
    print('training_user_cluster_CI95 =',d['positive_damage_cluster_ci95'])
    print('history_aware_minus_plain_mean_NLL =',format(d['mean_sequence_nll'],'.9g'))
    print('history_aware_minus_plain_RQ_MSE =',format(d['rq_mse'],'.9g'))
    print('step plain_H_after history_H_after')
    trajectories=[]
    for key in ('plain_dir','history_dir'):
        rows=[json.loads(line) for line in (Path(r[key])/'metrics.jsonl').read_text().splitlines() if line.strip()]
        trajectories.append({x['step']:x['history_after_optimizer_step']['positive_damage'] for x in rows if x.get('history_after_optimizer_step') is not None})
    for step in sorted(set(trajectories[0])&set(trajectories[1])):
        print(step,format(trajectories[0][step],'.9g'),format(trajectories[1][step],'.9g'))
    print('Training diagnostic only; no held-out improvement claim.')

if __name__=='__main__':main()
