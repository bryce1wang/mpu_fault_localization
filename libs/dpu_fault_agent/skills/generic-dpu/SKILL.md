---
name: generic-dpu
description: Diagnose generic DPU driver, firmware, initialization, reset, timeout, error, panic, or packet drop issues when no feature-specific skill matches.
feature: generic
module: dpu
problem_type: generic-triage
modules:
  - dpu
  - driver
  - firmware
keywords:
  - dpu
  - init
  - timeout
  - reset
  - error
symptoms:
  - fail
  - failed
  - timeout
  - drop
  - panic
required_evidence:
  - failing log window
  - DPU firmware and driver version
  - affected module or feature name
triage_steps:
  - Identify the failing stage and module from user symptoms or logs.
  - Compare firmware, driver, and configuration versions with a known-good setup.
  - Search source and configuration for the first concrete error code or module marker.
common_causes:
  - configuration mismatch between firmware and driver
  - initialization ordering issue
  - resource allocation or queue setup failure
validation_steps:
  - Collect a higher-verbosity DPU log around the failure window.
  - Reproduce on a known-good configuration and compare the first divergent event.
  - Inspect source near the first matched error code or module marker.
---

Use this generic skill when no feature-specific DPU skill matches. Keep conclusions conservative until concrete logs, source references, or reproduction details are available.
