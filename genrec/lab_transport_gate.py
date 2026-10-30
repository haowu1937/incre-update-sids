"""Task-local Git admission: fresh authorization, retrying I/O, no host upgrade."""
import json
import os
from pathlib import Path
import re
import subprocess
import time
from genrec.cf_letter_protocol import read, save


def redact(value):
    value=re.sub(r'(https?://)[^\s/@]+(?::[^\s/@]*)?@',r'\1[redacted]@',str(value))
    value=re.sub(r'(?i)(authorization\s*[:=]\s*)(?:basic|bearer|token)?\s*[^\r\n]+',r'\1[redacted]',value)
    value=re.sub(r'(?i)(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)', '[redacted]', value)
    value=re.sub(r'(?i)((?:password|access_token|token)=)[^\s&]+',r'\1[redacted]',value)
    return value[-8192:]


class GitGate:
    """None=communication wait, False=fresh permission, True=explicit pause."""
    def __init__(self,root,job_id,output):
        self.root=Path(root);self.job_id=job_id;self.output=Path(output)
        self.failures=0;self.next_retry=0.;self.last_success=None

    def check(self):
        if (self.root/'PAUSED.json').exists():return True
        if time.monotonic()<self.next_retry:return None
        config=read(self.root/'config.json');branch=config['requests_branch'];repo=str(self.root/'transport.git')
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_OPTIONAL_LOCKS='0',GCM_INTERACTIVE='Never')
        for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_TRACE','GIT_TRACE_CURL','GIT_CURL_VERBOSE'):env.pop(key,None)
        stage='fetch'
        try:
            subprocess.run(['git','--git-dir='+repo,'fetch','--quiet','--no-write-fetch-head','origin',
                '+refs/heads/'+branch+':refs/remotes/origin/'+branch],capture_output=True,check=True,timeout=45,env=env)
            stage='read_control'
            raw=subprocess.run(['git','--git-dir='+repo,'show','refs/remotes/origin/'+branch+':control/main.json'],capture_output=True,check=True,timeout=10,env=env).stdout
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            self.failures+=1;delay=(60,120,300,900)[min(self.failures-1,3)];self.next_retry=time.monotonic()+delay
            record=dict(status='waiting_for_git',stage=stage,failures=self.failures,retry_after_seconds=delay,
                returncode=getattr(error,'returncode',None),stderr=redact((getattr(error,'stderr',b'') or b'').decode(errors='replace') if isinstance(getattr(error,'stderr',b''),bytes) else error.stderr),
                time=time.time(),last_success=self.last_success)
            save(self.output/'control-transport.json',record)
            with (self.output/'control-transport.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True);return None
        # Malformed/ambiguous control is not an I/O failure; stop and diagnose.
        state=json.loads(raw)
        if type(state.get('pause_requested')) is not bool or not isinstance(state.get('active_job_id'),str):raise ValueError('Unknown remote authorization state')
        decision=state['pause_requested'] or state['active_job_id']!=self.job_id
        if not decision and state.get('mode') not in ('queued','running'):raise ValueError('Unexpected remote task mode')
        self.failures=0;self.next_retry=0.;self.last_success=time.time()
        save(self.output/'control-transport.json',dict(status='paused' if decision else 'connected',time=self.last_success,failures=0))
        return decision
