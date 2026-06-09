# DPU Fault Agent

A LangGraph-based CLI agent for visible and controllable DPU embedded software fault localization.

## Usage

```bash
dpu-fault-agent run --thread-id case-1 --problem "vf init timeout"
dpu-fault-agent run --thread-id case-2 --problem "rx drops" --log dpu.log --source src --skills ./skills
dpu-fault-agent status --thread-id case-1
dpu-fault-agent resume --thread-id case-1 --log new.log --source src --note "failure happens after queue setup"
dpu-fault-agent resume --thread-id case-1 --approve H1
dpu-fault-agent report --thread-id case-1 --output report.md
```

The agent accepts a problem statement first. Logs, source roots, and skill directories are optional and can be supplied at run time or later through `resume`.

## OpenCode

This repository includes a project-level OpenCode primary agent at `.opencode/agents/dpu_fault_agent.md`. OpenCode loads primary agents from `.opencode/agents/`, so the agent is available in the TUI agent cycle and can be selected with Tab.

The OpenCode agent is configured to use the `dpu_fault_agent` CLI through:

```powershell
D:\anaconda\python.exe -m dpu_fault_agent
```

It has read-only edit permissions by default and asks before running shell commands.

## Skills

Diagnostic skills are Markdown files with YAML front matter. The default directory is `skills/`; pass `--skills <dir>` to add feature-specific skills.

Required front matter fields:

- `id`
- `name`
- `modules`
- `keywords`
- `symptoms`
- `required_evidence`
- `triage_steps`
- `common_causes`
- `validation_steps`

The Markdown body is treated as feature-domain troubleshooting knowledge and is included in planning context.
