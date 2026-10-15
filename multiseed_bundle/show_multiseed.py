"""Three Update seed summary; standard library only."""
import json
from pathlib import Path
import sys

def pm(x):return format(x['mean'],'.9g')+' +/- '+format(x['sample_std'],'.4g')

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python show_multiseed.py THREE_SEED_SUMMARY_DIR')
    r=json.loads((Path(sys.argv[1])/'summary.json').read_text())
    if r.get('stage')!='three_seed_paired_update_summary_v1':raise ValueError('Unexpected summary')
    print('execution_status =',r['execution_status'],'| Update seeds =',r['seeds'],'| steps =',r['update_optimizer_steps'])
    print('Base_seed =',r['base_checkpoint_seed'],'| anchor_seed =',r['anchor_cohort_seed'],'| Base/teacher seeds varied =',r['base_or_teacher_seed_varied'])
    print('within_seed_same_order =',r['within_seed_same_order_all'],'| across_seed_different_orders =',r['across_seed_different_orders'])
    print('encoder = frozen | gate = disabled | rho =',r['design']['historical_v2']['history_gradient_ratio'])
    print('seed group plain_H history_H history-minus-plain')
    for s in r['per_seed']:
        for name,g in s['groups'].items():
            print(s['update_seed'],name,format(g['plain']['positive_damage'],'.9g'),format(g['history_aware']['positive_damage'],'.9g'),
                format(g['paired']['history_minus_plain_positive_damage'],'.9g'))
    print('Across seeds: mean +/- sample SD (ddof=1)')
    for name,g in r['groups'].items():
        h=g['positive_damage'];n=g['mean_delta_NLL']
        print(name,'plain_H =',pm(h['plain']),'| history_H =',pm(h['history_aware']))
        print('  H difference =',pm(h['history_minus_plain']),'| improved/worsened/tied seeds =',
            str(h['improved_seed_count'])+'/'+str(h['worsened_seed_count'])+'/'+str(h['tied_seed_count']))
        print('  mean_Delta_NLL difference =',pm(n['history_minus_plain']))
    print('RQ_MSE: plain =',pm(r['rq_mse']['plain']),'| history =',pm(r['rq_mse']['history_aware']))
    print('RQ_MSE history-minus-plain =',pm(r['rq_mse']['history_minus_plain']))
    print('holdout_used_for_training =',r['holdout_used_for_training'],'| ranking_evaluated =',r['ranking_evaluated'],'| same_compute =',r['same_compute'])
    print('Three Update sampling seeds share one fixed Base/teacher/cohort; this is not a full-pipeline multi-seed result.')

if __name__=='__main__':main()
