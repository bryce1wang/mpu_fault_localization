# DPU Fault Agent

A LangGraph-based CLI agent for visible and controllable DPU embedded software fault localization.

## Usage

```bash
dpu-fault-agent run --thread-id case-1 --problem "driver init fails" --log boot.log --source src
dpu-fault-agent status --thread-id case-1
dpu-fault-agent resume --thread-id case-1 --approve H1
dpu-fault-agent report --thread-id case-1 --output report.md
```

The MVP focuses on log and source-code evidence. It uses SQLite checkpoints so a run can pause at the human gate and resume later.
