"""Attachment wiring with the real config/deployment split; ML is mocked only here."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/lab_buffer'))
import attach_gradient_repair as attach


class AttachmentSchema(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'runner';self.root.mkdir()
        self.source=Path(self.tmp.name)/'source';(self.source/'scripts/lab_buffer').mkdir(parents=True)
        (self.root/'bin').mkdir();(self.root/'state').mkdir();(self.root/'scope-upgrade-backups').mkdir()
        self.names=('runner.py','main_control.py','main_service.py','gradient_logs.py','gradient_bridge.py')
        self.old={name:('legacy '+name+'\n').encode() for name in self.names}
        for name in self.names:
            (self.root/'bin'/name).write_bytes(self.old[name])
            (self.source/'scripts/lab_buffer'/name).write_bytes(self.old[name])
        (self.source/'scripts/lab_buffer/gradient_bridge.py').write_bytes(b'reviewed new gate\n')
        (self.root/'runner-code-sha.txt').write_text(attach.INSTALLED+'\n')
        (self.root/'config.json').write_text(json.dumps(dict(python=sys.executable,main_scope_revision=2)))
        # Actual attach_main.py schema: the interpreter lives in config.json.
        self.deployment=dict(schema_version=1,runner_root=str(self.root),input_sha256={})
        (self.root/'main-deployment.json').write_text(json.dumps(self.deployment))
        self.code='f'*40;self.events=[];self.commands=[];self.fail_preflight=False
        self.state=dict(active_job_id=attach.CONTROLLED,mode='needs_diagnosis',pause_requested=True,lease=None,
            extension=dict(scope_id=attach.SCOPE,logs_analysis_receipt={'verified':True},
                repair_delivery={'code_sha':self.code},dispatch_blocker={'kind':'old_gate'}))
        owner=self
        class Runner:
            def __init__(self,root):self.config=json.loads((Path(root)/'config.json').read_text())
            def ref(self,key):return key
            def read(self,ref,path):return None
            def publish(self,files):owner.events.append(('publish',files))
        class Control:
            def __init__(self,root):pass
            def read(self):return copy.deepcopy(owner.state),'parent'
            def claim(self,who):owner.state['lease']={'token':'fixture-token'};return 'fixture-token'
            def write(self,state,parent,extra=None):
                owner.state=copy.deepcopy(state);owner.events.append(('write',extra or {}))
        def git_output(argv,**kwargs):
            if argv[-2:]==['rev-parse','HEAD']:
                value=owner.code+'\n'
                return value if kwargs.get('text') else value.encode()
            if argv[-2]=='show':return owner.old[argv[-1].rsplit('/',1)[-1]]
            raise AssertionError(argv)
        def run(argv,**kwargs):
            if argv[0]=='git':return subprocess.CompletedProcess(argv,0)
            owner.commands.append(argv)
            owner.assertEqual(argv[0],sys.executable)
            owner.assertEqual(argv[-1],'--require-torch')
            kwargs['stdout'].write('Mocked ML subprocess for installer wiring only.\n')
            if owner.fail_preflight:raise subprocess.CalledProcessError(1,argv)
            return subprocess.CompletedProcess(argv,0)
        for p in (patch.object(attach,'ROOT',self.source),patch.object(attach,'Runner',Runner),
                  patch.object(attach,'Control',Control),patch.object(attach,'failure_evidence',return_value={}),
                  patch.object(attach.subprocess,'check_output',side_effect=git_output),
                  patch.object(attach.subprocess,'run',side_effect=run)):
            p.start();self.addCleanup(p.stop)

    def test_deployment_without_python_uses_runner_interpreter_and_reaches_queue(self):
        self.assertNotIn('python',self.deployment)
        result=attach.attach(self.root)
        self.assertEqual(result['status'],'attached')
        self.assertEqual(len(self.commands),1)
        self.assertEqual(self.state['active_job_id'],attach.CONTROLLED_REPAIR)
        self.assertFalse(self.state['pause_requested'])
        self.assertIsNone(self.state['lease'])
        request_files=[files for action,files in self.events if action=='write'][-1]
        request=json.loads(request_files['jobs/'+attach.CONTROLLED_REPAIR+'.json'])
        self.assertEqual(request['code_sha'],self.code)
        self.assertEqual((self.root/'bin/gradient_bridge.py').read_bytes(),b'reviewed new gate\n')
        self.assertEqual((self.root/'scope-upgrade-backups'/('positive-H-fixture-'+self.code)/'gradient_bridge.py').read_bytes(),self.old['gradient_bridge.py'])

    def test_failed_preflight_keeps_old_gate_and_does_not_submit_task(self):
        self.fail_preflight=True
        with self.assertRaises(subprocess.CalledProcessError):attach.attach(self.root)
        self.assertEqual(len(self.commands),1)
        self.assertEqual((self.root/'bin/gradient_bridge.py').read_bytes(),self.old['gradient_bridge.py'])
        self.assertEqual(self.state['active_job_id'],attach.CONTROLLED)
        self.assertTrue(self.state['pause_requested'])
        self.assertIsNone(self.state['lease'])
        self.assertFalse(any('jobs/' in path for action,files in self.events for path in files))


if __name__=='__main__':unittest.main()
