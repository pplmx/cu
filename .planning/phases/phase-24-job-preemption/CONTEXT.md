# Phase 24 Context: Job Preemption Handling

## Phase Overview

**Goal:** Graceful handling of cluster scheduler preemption.

## Requirements

| ID | Requirement | Success Criteria |
|----|-------------|------------------|
| PEMP-01 | Signal handlers | SIGTERM/SIGUSR1 handlers installed |
| PEMP-02 | State preservation | Graceful shutdown sequence |
| PEMP-03 | Resume validation | Checkpoint validation |
| PEMP-04 | Shutdown timeout | Configurable 30s default |
| PEMP-05 | Coordinated checkpoint | All ranks synchronized |

## Architecture

```
SignalHandler
├── install_handlers()
├── handle_signal()
└── trigger_shutdown()

ShutdownCoordinator
├── begin_graceful_shutdown()
├── preserve_state()
├── checkpoint_coordinated()
└── wait_for_completion()

ResumeValidator
├── validate_checkpoint()
├── recover_state()
└── verify_recovery()
```

## Gray Areas

1. **Timeout extension**: How to request more time from scheduler
2. **Graceful degradation**: How to save partial state if checkpoint fails
