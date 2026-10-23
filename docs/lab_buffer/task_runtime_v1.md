# Stable Runner / versioned task boundary

The Runner transports and executes trusted repository code. Experiment names, job
IDs, optimizer recipes, scientific manifests, checks and result verifiers belong
to the immutable task, not the installed Runner. A new experiment or diagnosed
repair using this v1 interface requires no Runner update or server command.
Only an actual transport/interface/host-capability change requires a core update.
Repository write access remains the trust boundary; this is not an OS sandbox.

## Host responsibilities

- Reuse the existing root, Python, cron, durable job claims, GPU wait and heartbeat.
- Fetch into named refs with `--no-write-fetch-head`; never consume shared FETCH_HEAD.
- Validate the exact request, contract, scope authorization, full code SHA and
  cumulative step reservations before a claim. Retain all old records.
- Create one detached worktree/output tree per unique job ID. Run declared CPU
  preflight checks, then the declared Python command, without shell evaluation.
- Preserve failure/interruption evidence. Deadlines pause new dispatch and leave
  running experiment processes intact. No retry, scope grant or auto-update.
- Verify returned hashes; invoke the scientific verifier from that same task SHA;
  append its English report under the existing CAS lease and commit the receipt.

The local `task-runtime-policy.json` describes supported protocol version, read
roots and maximum parallelism. It contains no experiment name, allowed job ID or
scientific budget. Drivers must enforce the declared step limit prospectively,
including rejected updates and uncertain intents, and respect pause/GPU rules.
The host checks reservations and reported actual usage; it is not an optimizer
instruction counter or a device/memory quota. Scientific verifiers audit those
driver counters and scientific invariants. Reviewed code must leave inputs intact;
hash checks detect changes but do not replace filesystem access controls.

## Git records and atomic dispatch

1. Read current protocol, state, capabilities and existing results. Require a
   reconciled completed previous plan, no pause and no lease. A failed plan needs
   an explicit evidence-backed reconciliation first. Never infer a new scope from
   a prior experiment's results. No task is authorized by installing this runtime.
2. Publish in **one CAS commit to lab-requests**:
   `task-scopes/SCOPE.json` (explicit user-approved immutable scope),
   `task-contracts/JOB.json`, `jobs/JOB.json`, and updated `control/main.json`.
   `task_runtime.queue` is the reference implementation; Web may perform the
   identical transaction through GitHub APIs without a server shell.
3. Set `task_dispatch={protocol_version:1, contract_path, contract_sha256}` and
   existing `active_job_id`, `code_sha`, `request_sha256`, `mode=queued` fields.
   Keep `task_scope_ledger[SCOPE]` with authorization hash, per-job reservations
   and actual steps. Total reservations cannot exceed the approved scope budget;
   unused reservations are not silently recycled. Preserve prior extensions.
4. The unchanged cron picks up the task. Manual and hourly checks observe the
   same job/state. Running/no new result means no second dispatch. Terminal
   verification writes one marker and receipt; subsequent processing is a no-op.
   Completion does not queue another experiment. The research controller may
   advance only a pre-approved remaining plan under the same CAS protocol.

Existing v1 request schema stays unchanged. `argv` is
`["python","-u","-B","relative_script.py", ...]`; each declared artifact is a
relative json/txt/md path, at most 64 KiB and eight artifacts. Always include
`outputs/task-runtime.json`; large traces/checkpoints stay in the local worktree.

Scope JSON fields:

```json
{
  "protocol_version": 1,
  "scope_id": "NEW-EXPLICITLY-APPROVED-SCOPE",
  "authorization": "User instruction and exact approved scientific plan identity",
  "max_optimizer_steps": 0,
  "max_parallel": 1
}
```

The zero budget above is an example, not a research task or approval. Contract
fields are exact; unknown fields/versions stop execution:

```json
{
  "protocol_version": 1,
  "scope_id": "SCOPE",
  "job_id": "UNIQUE-JOB",
  "code_sha": "FULL_REVIEWED_CODE_SHA",
  "request_sha256": "SHA256_OF_EXACT_REQUEST_BYTES",
  "scope_path": "task-scopes/SCOPE.json",
  "scope_sha256": "SHA256_OF_EXACT_SCOPE_BYTES",
  "manifest": {"path": "docs/experiments/scientific-plan.json", "sha256": "SHA256"},
  "preflight": [["python", "-u", "-B", "tests/task_checks.py"]],
  "verifier": "scripts/verify_this_task.py",
  "optimizer_step_reservation": 0,
  "input_sha256": {"/absolute/allowed/input/file": "SHA256"}
}
```

Inputs use exact paths and hashes within published host read roots. New parameters,
checks, code and verifiers are delivered at a new task SHA. Fixes use new job IDs;
an existing job, contract or scope authorization is never overwritten. Raising an
approved research budget requires a new user-approved scope, not a Runner change.

## Task code and verifier interface

The wrapper sets `LAB_BUFFER_JOB_ID`, `LAB_BUFFER_CODE_SHA`, `LAB_TASK_CONTRACT`,
`LAB_SCOPE_STEP_LIMIT` and `LAB_SCOPE_MAX_PARALLEL`. Checks run with CUDA hidden.
It rechecks pause before the experiment, hashes inputs before/after, validates
tracked source, and writes durable preflight/command/failure metadata. A preflight
failure starts zero experiment commands and returns its partial runtime artifact.
`training_started` in the transport result is a conservative process flag, not
evidence of optimizer steps; use driver counters and the verified receipt.

The verifier runs as a separate CPU process from the exact job worktree:
`python -u -B VERIFIER --context CONTEXT_JSON --output VERDICT_JSON`.
Context supplies request, contract, scope, result, workspace and output path.
The verifier must be read-only with respect to scientific inputs/results, perform
the task's declared scientific acceptance, and write a bounded JSON verdict:

```json
{
  "status": "verified",
  "job_id": "UNIQUE-JOB",
  "contract_sha256": "SHA256_OF_EXACT_CONTRACT_BYTES",
  "actual_optimizer_steps": 0,
  "report_markdown": "English facts, findings and evidence limits"
}
```

It does not claim a second lease or publish reports itself. The host checks artifact
hashes and actual-step bounds, then appends marker
`<!-- lab-task:JOB:CONTRACT_SHA256 -->`, publishes on the latest vibe-dev head using
CAS, and commits receipt/state together. A crash between report and receipt retains
the lease for reconciliation. Failure writes a needs_attention receipt (including
bounded verifier log tail when available) and pauses new work without a new job.

## One-time boundary migration

The installed 1bc0b43 executable has no generic job gate, so it cannot install this
interface by merely receiving a new experiment request. One explicit bootstrap
starts `install_task_runtime.py` in the background. It waits on the existing
Runner lock and for terminal delivery/controller receipt; it does not fetch against
the active parent's transport or alter its checkout. It runs local Git/subprocess
checks, validates exact prior helper bytes and an immutable upgrade authorization,
backs up owned files, and replaces only four core helpers plus runtime policy.
Legacy adapters stay available to process existing records. Main inputs, jobs,
receipts, user pauses, cron and running processes stay intact. Partial installation
leaves a durable pause and backup; no blind retry/rollback occurs. It submits zero
research tasks and spends zero Beauty optimizer steps.

After migration, every new task/repair must use the generic contract. Do not add a
new experiment-specific branch to `main_control.py`, install a bridge per study,
or ask Hao to attach a new job ID. The current 30,400-step study continues at its
original pinned SHA and is not converted, restarted or expanded by this change.
