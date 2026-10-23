# Proposed migration to collaborative SIDs and a LETTER-style retriever

Status: **design only; not authorized for execution**. Prepared on 2026-10-23.
No task is queued, no training/evaluation is started, and no previous budget is reused.
The completed content-RQ/TIGER experiments and their negative results remain intact.

## 1. Goal and interpretation of LETTER

Replace the current content-embedding RQ-VAE + T5/TIGER pipeline with:

**causal interactions -> SASRec item embeddings -> collaborative residual quantizer
-> four-token item SIDs -> decoder-only generative retriever.**

The target is the **Mitigating paper's LETTER-style adaptation**, not a claim to
reproduce the original LETTER objective. Original LETTER combines semantic,
collaborative and diversity regularization and can instantiate different retrievers.
Mitigating instead uses purely collaborative SASRec representations and a shared
decoder-only retriever. Its collaborative `Letter` class directly quantizes a frozen
embedding table; it does not contain the current MLP encoder/decoder. Name the new
line `CF-LETTER-adapted`, separate from the completed `Llama-TIGER` line.

Questions, in order:

1. Can the new Base pipeline learn useful collaborative SIDs and recommendation?
2. How much damage comes from refreshing embeddings versus updating the quantizer?
3. Can Hungarian alignment or the historical penalty preserve the old retriever?
4. After the same downstream update budget, which policy improves Future ranking?

Unlike static content features, a refreshed CF table can change old-item
representations from newly observed behavior. Measure that drift; do not assume
this change will resolve the earlier negative historical-penalty results.

Do not attribute an old-versus-new system difference solely to CF: backbone size,
history length, SID capacity and supervision also change.

## 2. Source audit and deliberate adaptations

References:

- [Mitigating Collaborative Semantic ID Staleness in Generative Retrieval,
  Sections 3.2 and 4.1](https://arxiv.org/html/2604.13273v1).
- [Author code, pinned at 133692ecd4ebf27f4d925a261952f8f6dec5d41c](https://github.com/iskbaga/semantic-id-alignment/tree/133692ecd4ebf27f4d925a261952f8f6dec5d41c).
- [Original LETTER, Sections 3.1–3.2](https://arxiv.org/html/2405.07314v3).

Important code details, checked against the pinned source:

| Component | Observed implementation | Proposed use |
| --- | --- | --- |
| Dense model | `dense-retriever/models.py`: SASRec, exported item lookup weights | Export item weights, not user-context states; no content vector concatenation |
| CF quantizer | `rqvae-collab/models.py`: direct residual quantization of `cf_embeddings`; no MLP encoder/decoder | Three learned 64-dimensional codebooks; identify this as CF-RQ |
| SID | Quantizer config has 3 learned books; export appends one collision token; retriever config has length 4 | Three semantic digits + one suffix; capacity 512 at each position |
| Decoder | `sid-retriever/models.py` sets `n_embd=embedding_dim*num_heads` | Config values 128 and 6 mean **actual d_model=768**, not 128 |
| Retriever loss | GPT causal-language-model CE in the inspected code | Standard CE; do not silently add original LETTER ranking/diversity losses |
| Author evaluation | Ten chronological blocks, fixed pre-Future context, old-item subset, high-cutoff metrics | Retain our existing five temporal splits and causal next-item queries; results are not a Table-2 numerical reproduction |

Additional adaptations must be explicit: warm-start SASRec refresh; deterministic
persistent suffixes; no dead-code reinitialization during the paired incremental
update; validation-based checkpoint selection within a common downstream cap.
The author scripts' default test-part evaluation is not used for checkpoint selection.

## 3. Data, catalog and availability

Reuse the existing immutable Beauty split/example identities and timestamp boundaries:
Base train / Base validation / Update train / Update validation / Future test.
Expected query counts remain 76,252 / 9,192 / 28,023 / 3,280 / 34,189 before the new
fixed-catalog availability masks. Retain the same 5-core provenance limitations.
Create a separate CF evaluation manifest; leave the original registries, features
and evaluation manifests read-only. Reuse event identities, not stale availability flags.
No same-timestamp event may enter a target's history. Truncate the most recent
**20 raw historical events before applying availability masks**; do not compact
unavailable blocks or rebuild leave-one-out samples.

The first main comparison uses a **common old-item catalog I0**, the 10,924 Base
train item identities. All arms train their downstream Update model on the same
eligible Update targets in I0 and rank the same I0. This follows the old-vocabulary
control in Mitigating and avoids confounding refresh with new-item coverage.

At Base, SASRec allocates only Base-observed item rows. At Update, append newly
observed identities without changing old rows/IDs. Positive loss targets remain
strictly in Base_train or Update_train. Earlier validation interactions may appear
only as the already-permitted causal input history, never as extra positive labels.
Negative samples use only the current training vocabulary and exclude the current
positive target. Never allocate future-only rows or sample future items as negatives.
Audit item exposure in training inputs, positive labels and negatives separately.

For quantizer refresh, use I0 plus new items with an allowed training input/positive
occurrence. Its size must be audited; **do not assume every one of the old 11,707
content-feature rows has a trained CF representation**. Keep low/zero-positive Base
rows in the fixed I0 comparison and disclose their exposure; do not select the test
catalog by observed model quality.

Base-validation-only items and the 68 Update-validation-first-seen items cannot be
given a meaningful new CF representation merely because a content vector exists.
Cold-start extension is outside this first comparison. Report new CF coverage,
not the old 80.56% content-pipeline coverage. Future summaries include both the
common served cohort and the full original 34,189-query denominator, counting
unavailable targets as misses. Report all-history-unavailable queries separately.

## 4. Proposed model settings

| Component | Initial configuration |
| --- | --- |
| SASRec | Item dimension 64; 2 causal layers; 2 heads; FFN 256; dropout 0.3; history 20 |
| SASRec fitting | AdamW, LR 1e-4, batch 256; sampled positive/negative BCE; explicit weight decay 0.01; export raw FP32 item lookup weights |
| CF-RQ | Input/quantized dimension 64; identity encoder/decoder; 3 x 512 learned books; commitment coefficient 0.25; outer quantization weight 1.0 |
| Quantizer fitting | AdamW, LR 1e-4, weight decay 0.01, batch 4096; exact source loss and repair counts logged |
| Collision suffix | Fourth positional token, fixed capacity 512; deterministic versioned ledger; no random reassignment or silent expansion |
| Generative retriever | Randomly initialized GPT-style decoder; actual d_model 768, 4 layers, 6 heads, head dimension 128, FFN 1024, ReLU, dropout 0.1 |
| Vocabulary | Four disjoint 512-token positional ranges, 8,000 stable user-hash IDs and explicit special tokens; save the full token manifest |
| Retriever fitting | AdamW, LR 1e-4, weight decay 0.01, batch 256, gradient norm cap 1.0, constant LR; no pretrained download |
| Precision | FP32 first; any mixed-precision variant requires a separate recorded change |

Each existing explicit causal next-item query is used once per epoch. For SASRec,
supervise its last valid history representation for that one target. For the
decoder, concatenate user token + masked history SIDs + target SID, and supervise
**only the four target digits**. Context/user/padding labels are ignored. Do not
silently expand every existing query into all prefixes. This target-only adapter
differs from the author decoder's all-position training and preserves our event
weighting and temporal protocol. Causal target teacher forcing must be tested.
Generation stops after four digits; no target EOS is added in this new protocol.
Use explicit, consistent position IDs for raw SID positions in training, historical
scoring and cached generation. Interior masked blocks must not cause an accidental
attention-cumsum position renumbering in GPT generation.

The direct CF quantizer needs a loss/gradient audit before implementation is frozen:
the inspected source uses a frozen CF buffer and detached hard-vector STE, so its
reported reconstruction term must not be assumed to train an absent encoder or
to send reconstruction gradients into every codebook. Preserve its forward objective
and document which terms actually carry gradients. Log the source objective, hard
reconstruction MSE and each residual/commitment term separately. Do not compare its
absolute loss to the old 4096-dimensional autoencoder objective.

## 5. Base and shared CF refresh

1. Fit SASRec S0 on Base_train; select its checkpoint on Base validation only,
   with a declared NDCG@10 rule. Export E0 with item-row, source and checkpoint hashes.
2. Train CF quantizer C0 on E0. Save complete hard m0 assignments, suffix ledger,
   per-book utilization and collisions. Fit the new Base retriever M0 from scratch
   using m0; select on a fixed Base-validation probe, then evaluate full validation.
   Do not reuse a T5 checkpoint or old token embedding table in this new GPT model.
3. Warm-start SASRec S1 from the selected S0 and fit the cumulative **Base_train +
   Update_train** query set for a fixed 30-epoch cap, with reset AdamW and stable old
   row initialization. Export E1 once and freeze it for every refresh arm. This
   limits arbitrary coordinate rotations relative to independently trained CF
   models; it does not remove genuine embedding drift. From-scratch S1 is a later
   sensitivity study, not an extra factor in the first grid.

At every stage, check row identity, norms, zero vectors, repeated-item counts and
train/validation boundaries. Use SASRec's own old-catalog Recall/NDCG as a dense
reference; it is not a theoretical upper bound on the generative model.

## 6. Experiment matrix

Every frozen downstream condition uses the exact M0. Every fine-tuned condition
warm-starts the exact M0 with a reset optimizer and identical query order/budget.

| Arm | Input representation / tokenizer | Purpose | Downstream conditions |
| --- | --- | --- | --- |
| Base / FT-old | Persist m0=C0(E0); no old-item re-encoding | Frozen-SID reference, analogous to FT-old | Frozen once; then Update fine-tuning |
| Feature-only diagnostic | C0 weights frozen, apply E1 | Separate embedding refresh damage from quantizer training | Frozen-teacher H only; no extra Future arm |
| FT-new | Fresh Cnew trained on E1, raw refreshed SIDs | Paper-style unaligned rebuild reference | Frozen + Update fine-tuning |
| FT-align | Exactly the same Cnew checkpoint as FT-new, followed by Hungarian token alignment | Mapping-only control analogous to FT-ours | Frozen + Update fine-tuning |
| Plain incremental | Warm-start C0, fit E1 with quantizer objective only | Matched control for our historical method | Frozen + Update fine-tuning |
| Historical incremental | Same C0/E1, objective L_Q + lambda H | Test historical compatibility under collaborative drift | Frozen + Update fine-tuning |

This produces five distinct mapping arms and at most ten distinct Future
mapping/model conditions; reuse identical frozen Base/FT-old predictions. Feature-only
is an anchor diagnostic, not a sixth Future mapping. The expensive Full retrain
reference and content-SID/same-GPT bridge control are deferred, separately budgeted
studies; do not call the current matrix a full reproduction of every paper baseline.

The feature diagnostic is essential: **frozen quantizer weights do not mean frozen
SIDs when CF embeddings change**. Record H(m0), H(C0(E1)), H(Plain(E1)) and
H(Historical(E1)). The paired optimization starts at C0(E1), whose H may already be
positive. Do not replace the old reference loss with that post-refresh starting loss.

For Hungarian, use unweighted old/new assignment overlap on I0, with a bijection
within each of the four positional token ranges, including the suffix; complete
unused-token matches deterministically. Apply the stored permutations to the entire
new mapping and resolver. Do not permute the old retriever weights, special tokens
or target references in historical diagnostics. No suffix regeneration after the
bijection: uniqueness is preserved. Matching sees item assignments, not anchor
holdout losses, validation scores or Future labels.

## 7. Historical loss, calibration and monitoring

Use a frozen/eval M0 and the old full target SID y0:

`H = mean_q relu(NLL(M0, new_history_q, y0_q) - NLL(M0, old_history_q, y0_q))`.

NLL is the **sum over four target digits**, using the ordinary model softmax;
catalog-constrained generation is only for ranking. Also report per-token values.
This differs from the previous five-token T5 target (four digits + EOS), so do not
carry lambda over by numerical coincidence. SASRec/E1 and M0 stay frozen during
all quantizer updates. The new Historical variant updates the learned codebooks;
there is no MLP encoder/decoder to update.
Expose a separate teacher-forced scoring API accepting `inputs_embeds` while M0
stays in eval mode: the author wrapper's eval branch calls generation and cannot
serve as the differentiable historical loss. Cache only the old-reference branch
under no_grad; retain autograd through the new-history input branch. Verify shifted
four-digit labels, dropout-off ID/embedding parity and per-level gradient routes.

Retain the existing 4,096 anchor_train and 4,096 anchor_holdout identities, re-encoded
under the new 20-event protocol. The holdout has been inspected before and is not
a new blind test. Create up to 1,024 additional anchor_dev queries from remaining
Base_train users, disjoint from both cohorts; freeze identities/counts before pilots.
Only anchor_train enters the penalty. Only anchor_dev/training diagnostics choose
lambda. Base/Update ranking validation and Future do not choose historical settings.

Initial history settings: anchor batch 128, distance-STE temperature 0.5, Base-frozen
distance scales and gradient clip 0.1, identical in paired Plain/Historical runs.
No incremental dead-code reset or token alignment in this pair. Base/from-scratch
quantizer initialization/repair follows a separately recorded fixed policy.

Budget a 200-step Plain calibration and three 200-step Historical pilots. On positive-H
training probes, estimate a single fixed scale
`lambda0 = 0.1 * median(||g_Q|| / ||g_H||)` over learned codebooks; test
`lambda0/3`, `lambda0`, `3*lambda0`. This is a one-time training calibration, not a
dynamic gradient-normalized objective. Skip zero-history-gradient probes; if all
are zero, stop that stage and diagnose rather than manufacture a ratio. Select
the lowest anchor_dev hard H among capacity-feasible candidates whose hard
reconstruction is within 2% of matched Plain, with declared deterministic ties.
If none is feasible, stop the Historical stage; no automatic larger sweep.

Then run a fresh matched Plain/Historical pair for 1,000 updates each. Report true
pre-update batch H separately from fixed-cohort curves. Every 10 complete content
epochs, and at step 0/final, evaluate full fitting-catalog hard MSE/source objective,
full anchor_train, anchor_dev, utilization, semantic/full-SID churn and suffix capacity.
Check capacity after every optimizer step; retain rejected candidates and last valid
state, stop without expansion, and count every committed update.

Predeclare the last eight scheduled complete-epoch monitors for stability: RQ
relative drift <=2%, RQ CV <=3%, H relative drift <=2%, H range <=10% of its mean;
report zero-H plateaus separately. These are finite-window criteria. A useful
historical setting additionally requires >=5% lower hard H than Plain on the
training/development comparison with acceptable reconstruction. After settings
are locked, report paired user-cluster bootstrap intervals on anchor_holdout.
Do not choose a checkpoint from that holdout or suppress negative results.

## 8. Downstream selection and Future

Retain the existing Update selection protocol: train every FT arm for at most
2,000 steps; candidates 0/100/250/500/1000/2000; fixed 512-query Update-validation
probe; full-denominator NDCG@10; earliest tie. Actual training and selected steps
are separate report fields. Full Update validation is evaluated on the selected
checkpoint. The common I0 availability mask and probe IDs are identical across arms.

After all configurations, mappings and checkpoints are locked, perform Future
evaluation with the existing **prequential** raw-history rule, max 20 raw events,
fixed I0 catalog, no parameter/CF/mapping update and no seen-item filtering.
Use the current arm's same mapping for history and target in real recommendation;
new-history/old-target is only the historical-damage diagnostic.

First-stage ranking metrics: Recall/NDCG@5/@10/@20, beam 20; report full and common
served denominators, coverage, old-item frequency slices and invalid outputs. This
does **not** reproduce the paper's fixed-history, per-user multi-positive @500
protocol. A high-cutoff / fixed-history / ten-block reproduction is a separate
later protocol, not a silent change to these Future queries. No Future result may
change lambda, CF fitting, step caps, beam width or the candidate set.
New Future rows belong to a clearly separated protocol block in the primary
downstream table: different catalog, backbone, history length and target length
must remain visible alongside the old Llama/TIGER results.

## 9. Proposed budget, gates and automation

One engineering seed (2026) first; it cannot support a cross-seed effectiveness claim.
Three full independent pipeline seeds would require a separate confirmation budget;
do not mistake multiple Update-only seeds for full pipeline replication.

| Stage | Proposed maximum |
| --- | --- |
| SASRec Base | 150 epochs, <=44,700 steps for 76,252 queries/batch 256 |
| Base CF-RQ | 1,000 steps |
| Base GPT | 30 epochs, <=8,940 steps on the same causal queries |
| Shared SASRec refresh | 30 cumulative epochs, <=12,240 steps for 104,275 queries |
| Fresh CF-RQ / Plain / Historical | 1,000 steps each, <=3,000 total |
| Lambda pilots, including Plain | 4 x 200 =800 steps |
| Five downstream FT arms | 5 x 2,000 =10,000 steps |
| Optional throughput calibration within this cap | <=200 steps, recorded separately |
| **Proposed total** | **<=80,880 optimizer steps across different model types** |

These heterogeneous steps are not equivalent compute units and are not comparable
to the previous 30,400-step RQ-only budget. Propose an additional device-time ceiling
of 12 V100-hours for fitting/checks/validation and 8 V100-hours for final inference,
subject to measured profiling and Hao's approval. A deadline stops new dispatch;
already-running experiments finish. No automatic budget increase, retry, extra seed,
Full retrain, content bridge or @500 evaluation. Unused previous 9,000 steps remain
unused. Report both actual shared hardware cost and method-attributed cost including
CF fitting, tokenization, H, alignment, re-encoding, trie and retriever fitting.

Order: integrity/gradient tests -> Base pipeline acceptance -> shared CF refresh and
feature-only diagnostic -> raw rebuild/alignment and paired historical pilots/update
-> identical downstream fitting -> locked Future evaluation/report -> stop.
If a technical gate fails, preserve evidence and diagnose; do not automatically
scan parameters to repair a scientific negative result.

Use the already installed generic task runtime, unique job IDs, full code/manifest
SHAs, independent outputs and immutable contracts. No per-experiment Runner upgrade
or manual result copying. Execution requires a newly approved scope/budget record;
this design document creates none. Keep the completed current control state intact.
The first approved execution should be one bounded top-level bundle, with unique
per-stage/per-arm child invocation IDs and separate subdirectories. Read the
existing temporal input root `cache/incremental/Beauty/beauty_20261007T104452Z`
through exact captured file hashes. Keep all new CF tables, Base models, branches
and dependencies inside that job's independent Runner workspace, with internal
hash/lineage checks. Do not write new outputs into the main experiment checkout
or require a new host read-root for intermediate artifacts.

Return bounded `compact.json`, `validation.json`, `anchors.json` and `curves.json`
plus the mandatory runtime artifact; each is <=64 KiB. Full checkpoints, CF tables,
query-level Future predictions, gradients and logs remain in the unique workspace.
The task-SHA scientific verifier rechecks temporal/mapping identities, counters,
checkpoint reloads and metrics aggregated from saved query records before the
generic host publishes its report/receipt. Failed stages preserve partial evidence
and stop subsequent dispatch; do not rerun the whole bundle automatically.

Minimum checks before real work: causal labels/timestamps; stable old item rows;
no Future-only CF rows or negatives; frozen E1/teacher; identical ID-vs-embedding
hard logits; same-map H=0; post-CF-refresh H measured without assuming zero;
per-level historical gradient routes; hard-SID H after actual updates; full-suffix
feasibility; four-token causal target masking; permutation/resolver consistency;
sampler/optimizer/RNG resume and request/result/receipt deduplication.
