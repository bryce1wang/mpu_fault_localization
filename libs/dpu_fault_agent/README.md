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

Diagnostic skills use directory-based `SKILL.md` files with YAML front matter. The default directory is `skills/`; pass `--skills <dir>` to add feature-specific skills.

Recommended layout:

```text
skills/
|-- generic-dpu/
|   `-- SKILL.md
|-- virtualization/
|   `-- vf/
|       `-- vf-init-timeout/
|           |-- SKILL.md
|           |-- scripts/
|           |   `-- collect.py
|           `-- references/
|               `-- evidence.md
`-- networking/
    `-- rx/
        `-- rx-drop/
            `-- SKILL.md
```

The recommended skill granularity is one module problem type, such as `vf-init-timeout` or `rx-drop`.

Required front matter fields:

- `name`
- `description`
- `feature`
- `module`
- `problem_type`
- `keywords`
- `symptoms`
- `required_evidence`
- `triage_steps`
- `common_causes`
- `validation_steps`

Optional `modules` can include aliases for matching. If omitted, the primary `module` is used.

Optional `scripts` declares Python scripts that the agent should run after the skill is matched:

```yaml
scripts:
  - name: collect
    path: scripts/collect.py
    args: ["--fast"]
    timeout_seconds: 30
```

Script paths must point to `.py` files inside the skill directory. They run with the Python interpreter used by the agent, with the skill directory as the working directory. Script output is captured as `skill_script` observations for planning and reporting. The agent sets these environment variables for scripts:

- `DPU_FAULT_AGENT_THREAD_ID`
- `DPU_FAULT_AGENT_PROBLEM`
- `DPU_FAULT_AGENT_SKILL_ID`
- `DPU_FAULT_AGENT_SKILL_NAME`
- `DPU_FAULT_AGENT_FEATURE`
- `DPU_FAULT_AGENT_MODULE`
- `DPU_FAULT_AGENT_PROBLEM_TYPE`
- `DPU_FAULT_AGENT_LOG_PATHS`
- `DPU_FAULT_AGENT_SOURCE_ROOT`

`SKILL.md` body should stay concise: include when to use the skill, the diagnostic workflow, evidence interpretation rules, and escalation criteria. Longer error-code tables, sample logs, or module diagrams should live in `references/`.

Legacy flat `skills/*.md` files are still loaded for migration, but new skills should use directory-based `SKILL.md` files.
