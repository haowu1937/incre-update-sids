"""Pure distance/protection checks and actual CPU checkpoint-index diagnosis."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
from audit_cf_letter_optimizer import evidence,check_source,inspect_checkpoint
from genrec.cf_letter_protocol import save,sha
HAVE=importlib.util.find_spec('torch') is not None


class Protocol(unittest.TestCase):
    def test_distance_evidence_distinguishes_tie_and_wrong_choice(self):
        x=np.array([[0.,0.],[2.,0.]],dtype=np.float32);books=[np.array([[-1.,0.],[1.,0.]],dtype=np.float32)]
        stored=np.array([[1],[0]]);cpu=np.array([[0],[1]])
        rows=evidence(x,books,stored,cpu,np.array([[0,0],[1,0]]))
        self.assertEqual(rows[0]['saved_minus_nearest_float64'],0.);self.assertEqual(rows[1]['saved_minus_nearest_float64'],8.)

    def test_source_index_rejects_changed_files_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'compact.json',{'original':True});(root/'weights.npy').write_bytes(b'original')
            save(root/'local_artifact_index.json',{'weights.npy':sha(root/'weights.npy')});p=dict(source_compact_sha256=sha(root/'compact.json'),source_index_sha256=sha(root/'local_artifact_index.json'))
            self.assertEqual(check_source(root,p),1);(root/'weights.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):check_source(root,p)

    @unittest.skipUnless(HAVE,'Torch unavailable; actual CPU checkpoint audit required on server')
    def test_real_checkpoint_audit_preserves_saved_codes_and_records_mismatch(self):
        import torch
        from genrec.cf_letter_models import CollaborativeRQ
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=dict(dim=2,books=3,capacity=2,commitment=.25,outer_weight=1.)
            model=CollaborativeRQ(torch.tensor([[0.,0.],[2.,0.]]),cfg)
            with torch.no_grad():
                model.books[0].copy_(torch.tensor([[-1.,0.],[1.,0.]]))
                for book in model.books[1:]:book.copy_(torch.tensor([[0.,0.],[10.,10.]]))
            codes=model.assignments().numpy();np.save(root/'codes.npy',codes);np.save(root/'old.npy',codes)
            torch.save(dict(model=model.state_dict(),settings={'quantizer':cfg}),root/'checkpoint.pth')
            before=sha(root/'checkpoint.pth');metrics,_=model.audit();value,_=inspect_checkpoint(root/'checkpoint.pth',root/'codes.npy',root/'old.npy',metrics['reconstruction_mse'])
            self.assertEqual(value['cpu_mismatch_elements'],0)
            changed=codes.copy();changed[0,0]=1-codes[0,0];np.save(root/'codes.npy',changed)
            value,samples=inspect_checkpoint(root/'checkpoint.pth',root/'codes.npy',root/'old.npy',metrics['reconstruction_mse'])
            self.assertEqual(value['cpu_mismatch_elements'],1);self.assertEqual(samples[0]['saved_minus_nearest_float64'],0.);self.assertEqual(sha(root/'checkpoint.pth'),before)


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:sys.argv.remove('--require-torch')
    if require and not HAVE:raise SystemExit('Actual Torch checkpoint audit check required')
    unittest.main()
