"""Exact joint-study gate and terminal evidence acceptance; never expands the grid."""
import json
from pathlib import Path
import sys

from runner import digest,now
from main_control import encoded

SCOPE='history-joint-convergence-v1'
JOB='history-joint-convergence-001'
ARTIFACTS=['outputs/convergence/'+n for n in
    ('compact.json','screen-curves.json','extended-curves.json','confirmation-curves.json')]


def task(root,code,created_at=None):
    return dict(schema_version=1,job_id=JOB,worker_id='server2',kind='experiment',code_sha=code,
        created_at=created_at or now(),timeout_seconds=86400,
        argv=['python','-u','-B','run_joint_convergence.py','--deployment',str(Path(root)/'convergence-deployment.json'),
              '--output','outputs/convergence'],artifacts=ARTIFACTS)


def authorize(root,request,raw,state):
    ext=state.get('extension',{})
    if (state['pause_requested'] or state['mode']!='queued' or ext.get('scope_id')!=SCOPE
            or ext.get('status')!='queued' or state['active_job_id']!=JOB):return False
    d=json.loads((Path(root)/'convergence-deployment.json').read_text())
    if (state['code_sha']!=request['code_sha'] or state['request_sha256']!=digest(raw)
            or d['code_sha']!=request['code_sha'] or ext['max_optimizer_steps']!=30400
            or ext.get('real_model_preflight_passed') is not True
            or request!=task(root,state['code_sha'],request['created_at'])):
        raise ValueError('Joint study request is outside its exact command/SHA/preflight/budget')
    return True


def validate_artifacts(r,request,result):
    if (result['status']!='completed' or result['exit_code']!=0
            or result.get('actual_code_sha')!=request['code_sha']):raise ValueError('Study job did not complete normally')
    expected=['artifacts/'+JOB+'/'+name for name in ARTIFACTS]
    if result['artifacts']!=expected:raise ValueError('Study artifact list mismatch')
    values={}
    for relative,path in zip(ARTIFACTS,expected):
        raw=r.read(r.ref('results_branch'),path)
        if raw is None or digest(raw)!=result['artifacts_sha256'].get(path):raise ValueError('Study artifact digest mismatch')
        values[Path(relative).name]=json.loads(raw)
    report=values['compact.json']
    if (report['stage']!=SCOPE or report['execution_status']!='completed' or report['job_id']!=JOB
            or report['code_sha']!=request['code_sha'] or report['max_optimizer_steps']!=30400
            or not 400<=report['optimizer_steps_spent']<=30400 or report.get('uncertain_step_records')
            or report.get('original_inputs_unchanged') is not True
            or any(report.get(key) is not False for key in ('holdout_evaluated','base_validation_evaluated','future_evaluated'))
            or report['new_tiger_training_steps']!=0):raise ValueError('Study evidence violates the train-only protocol')
    for name in ('screen-curves.json','extended-curves.json','confirmation-curves.json'):
        path='artifacts/'+JOB+'/outputs/convergence/'+name
        if report['curves_sha256'][name]!=result['artifacts_sha256'][path]:raise ValueError('Curve binding differs')
    return report,values


def report_text(report,marker):
    lines=['',marker,'### Joint RQ/H convergence study: server result','',
        'Train-only study; no holdout, ranking or Future scoring. Immutable job `'+JOB+'`, code `'+report['code_sha']+'`.',
        'Actual new optimizer steps: '+str(report['optimizer_steps_spent'])+' / 30400. Outcome: `'+report['research_outcome']+'`.',
        'Selected concurrency: '+str(report['calibration']['selected_parallelism'])+'.',
        'The table reports fixed-budget endpoints. Selection and stability use the declared eight-epoch training windows.',
        '', '| Phase | LR | lambda | Seed | Step | Full RQ loss | Anchor train H | Status |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for phase in ('screen','extended','confirmation'):
        for arm in report['phases'].get(phase,[]):
            c=arm['case'];e=arm.get('endpoint',{})
            rq=e.get('rq',{}).get('total_loss');hs=e.get('H')
            lines.append('| '+ ' | '.join([phase,str(c['lr']),str(c['history_weight']),str(c['seed']),
                str(arm.get('valid_step',0)),format(rq,'.9g') if rq is not None else '—',
                format(hs,'.9g') if hs is not None else '—',arm['status']])+' |')
    lines.extend(['','Outcome is training-side evidence only. Existing downstream/Future results remain unchanged.',
        'Full curves, checkpoints, resource observations and failures remain in the independent Runner workspace; '
        'bounded curve tables and compact evidence are returned through lab-results.',''])
    return '\n'.join(lines)


def observe(control,state):
    ext=state['extension'];r=control.runner
    if state['pause_requested']:return dict(status='paused')
    if ext['status'] in ('completed','failed'):return dict(status=ext['status'])
    raw=r.read(r.ref('requests_branch'),'jobs/'+JOB+'.json')
    result_raw=r.read(r.ref('results_branch'),'jobs/'+JOB+'.json')
    if result_raw is None:return dict(status='waiting',job_id=JOB)
    result=json.loads(result_raw);request=json.loads(raw)
    if digest(raw)!=state['request_sha256'] or result['request_sha256']!=digest(raw) or any(result[k]!=request[k] for k in ('job_id','code_sha','worker_id','kind')):
        raise ValueError('Joint study result/request identity mismatch')
    if result['status'] in ('starting','running'):return dict(status=result['status'],job_id=JOB)
    token=control.claim('server-joint-study-acceptance')
    if token is None:return dict(status='controller_busy')
    state,parent=control.read()
    try:
        report,artifacts=validate_artifacts(r,request,result)
        # Verify arithmetic and selection using the exact SHA-pinned workspace code.
        source=Path(result['workspace']).resolve()
        if source!=(Path(control.root)/'workspaces'/JOB).resolve():
            raise ValueError('Unexpected study workspace; do not import result-provided code')
        for relative in ('joint_convergence.py','run_joint_convergence.py'):
            expected=r.git('show',request['code_sha']+':'+relative).stdout
            if digest((source/relative).read_bytes())!=digest(expected):
                raise ValueError('Study verifier source differs from pinned Git object')
        sys.path.insert(0,str(source))
        import joint_convergence as protocol
        from run_joint_convergence import expected_steps
        actual,uncertain=expected_steps(source/'outputs/convergence')
        if actual!=report['optimizer_steps_spent'] or uncertain:raise ValueError('Returned budget differs from local durable counters')
        local=json.loads((source/'outputs/convergence/summary.json').read_text())
        if protocol.sha(source/'outputs/convergence/summary.json')!=report['full_summary_sha256']:
            raise ValueError('Local full summary changed')
        phase_summaries={}
        for phase in ('screen','extended','confirmation'):
            target=1500 if phase=='screen' else 3000
            phase_summaries[phase]=[json.loads((source/'outputs/convergence/arms'/row['output_id']/('summary-'+str(target)+'.json')).read_text())
                for row in local['phases'].get(phase,[]) if row.get('status') in ('completed','stopped_capacity_infeasible')]
            for s in phase_summaries[phase]:
                if s['status']=='completed':
                    cp=source/'outputs/convergence/arms'/s['case']['arm_id']/('checkpoint-'+str(target)+'.pth')
                    if protocol.sha(cp)!=s['archived_checkpoint_sha256']:
                        raise ValueError('Archived stage checkpoint digest differs')
        if protocol.select_screen(phase_summaries['screen'])!=report['screen_selection']:
            raise ValueError('Screen selection arithmetic does not reproduce')
        if protocol.select_final(phase_summaries['extended'])!=report['final_selection']:
            raise ValueError('Final selection arithmetic does not reproduce')
        if report.get('confirmation') and protocol.select_final(phase_summaries['confirmation'])!=report['confirmation']:
            raise ValueError('Confirmation criteria do not reproduce')
        expected_cases=protocol.matrix(protocol.read_plan(source))
        if sorted(s['case']['arm_id'] for s in phase_summaries['screen'])!=sorted(s['arm_id'] for s in expected_cases):
            raise ValueError('Screen did not account for every declared arm')
        if sorted(s['case']['arm_id'] for s in phase_summaries['extended'])!=report['screen_selection']['arm_ids']:
            raise ValueError('Extension arm set differs from the predeclared screen')
        if any(s['target']!=1500 for s in phase_summaries['screen']) or any(s['target']!=3000 for s in phase_summaries['extended']+phase_summaries['confirmation']):
            raise ValueError('Unexpected stage endpoint')
        artifact_hash=result['artifacts_sha256']['artifacts/'+JOB+'/outputs/convergence/compact.json']
        marker='<!-- joint-convergence:'+artifact_hash+' -->'
        r.fetch(('code_branch',));head=r.git('rev-parse',r.ref('code_branch')).stdout.decode().strip()
        content=r.read(r.ref('code_branch'),'experiment_report.md').decode()
        if marker not in content:r.publish({'experiment_report.md':(content.rstrip()+'\n'+report_text(report,marker)).encode()},'code_branch',expected_parent=head)
        r.fetch(('code_branch',));report_sha=r.git('rev-parse',r.ref('code_branch')).stdout.decode().strip()
        state,parent=control.read()
        if (state.get('lease') or {}).get('token')!=token:raise ValueError('Lost report acceptance lease')
        receipt=dict(scope_id=SCOPE,job_id=JOB,status='verified_and_reported',code_sha=request['code_sha'],
            request_sha256=digest(raw),artifact_sha256=artifact_hash,report_commit_sha=report_sha,
            actual_optimizer_steps=actual,processed_at=now(),next_job_id=None,
            evidence='Server verified local counters, bound curves and training-only selection arithmetic; no independent cloud model recomputation.')
        state['extension'].update(status='completed',result_receipt=receipt,optimizer_steps_spent=actual)
        state.update(mode='plan_completed',lease=None)
        control.write(state,parent,{'phase-receipts/'+SCOPE+'/'+JOB+'.json':encoded(receipt)})
        return dict(status='plan_completed',receipt=receipt)
    except BaseException as error:
        state,parent=control.read()
        if (state.get('lease') or {}).get('token')==token:
            state.update(mode='needs_diagnosis',pause_requested=True,lease=None)
            state['extension'].update(status='failed',failure_reason=repr(error),failed_job_id=JOB)
            control.write(state,parent)
        raise
