"""Accept all40000 family updates, charge only6000 new updates to this job."""
import argparse
from pathlib import Path
from genrec.cf_letter_protocol import read,save,sha
from genrec.cf_letter_multiseed_protocol import gpt_cases
from genrec.cf_letter_multiseed_resume_protocol import JOB,SCOPE,SOURCE_CODE,MISSING,plan,source_check,check_reused,expected_counts
from run_cf_letter_multiseed import total_counters
ROOT=Path(__file__).resolve().parent


def verify(context):
    c=read(context);m,p=plan(ROOT);root=Path(c['workspace']).parent.parent;output=Path(c['workspace'])/'outputs/cf-multiseed-resume'
    source,index=source_check(root,m,p);check_reused(output)
    if read(output/'provenance/source_index.json')!=index or read(output/'provenance/source_compact.json')!=read(source/'compact.json'):
        raise ValueError('Source provenance differs')
    reused=read(output/'provenance/reused_files.json')
    from genrec.cf_letter_multiseed_resume_protocol import immutable_reuse
    if reused!={n:v for n,v in index.items() if immutable_reuse(n)}:raise ValueError('Missing copied source evidence')
    r=read(output/'compact.json');total,counts,intents=total_counters(output)
    new=sum(counts.get('gpt/'+n,0) for n in MISSING)
    if (intents or counts!=expected_counts(p,True) or total!=40000 or new!=6000 or r['new_optimizer_steps_spent']!=new
        or r['inherited_optimizer_steps']!=34000 or r['new_optimizer_step_cap']!=6000):raise ValueError('Continuation budget differs')
    codes={case['name']:c['request']['code_sha'] if case['name'] in MISSING else SOURCE_CODE for case in gpt_cases(p)}
    from verify_cf_letter_multiseed import verify as verify_family
    verdict=verify_family(context,configuration=p,directory=output,job_id=JOB,scope_id=SCOPE,rq_code_sha=SOURCE_CODE,gpt_code_shas=codes)
    if verdict['actual_optimizer_steps']!=40000:raise ValueError('Family verification incomplete')
    source_check(root,m,p);check_reused(output)
    verdict.update(actual_optimizer_steps=6000,report_markdown=verdict['report_markdown']+'\n\nContinuation accounting: original failed job retained34000 updates; this job performed only6000 new GPT updates. Family actual total40000, no repeated codebook or earlier GPT training. All15 Future passes follow one new global lock. Original failure and records remain unchanged.')
    return verdict


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--context',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();save(a.output,verify(a.context))
