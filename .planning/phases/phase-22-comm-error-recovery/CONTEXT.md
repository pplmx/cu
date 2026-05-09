# Phase 22 Context: Communication Error Recovery

## Phase Overview

**Goal:** Detect and recover from NCCL/TCP communication failures.

## Requirements

| ID | Requirement | Success Criteria |
|----|-------------|------------------|
| COMM-01 | NCCL timeout detection | Configurable thresholds (default 60s) |
| COMM-02 | Health monitoring | Watchdog thread for collectives |
| COMM-03 | Automatic retry | Exponential backoff (max 3 attempts) |
| COMM-04 | Connection repair | Communicator recreation |
| COMM-05 | Error classification | Transient vs permanent decision |

## Architecture

```text
CommErrorRecovery
├── HealthMonitor (watchdog thread)
│   ├── periodic_check()
│   └── stall_detection()
├── RetryHandler
│   ├── exponential_backoff()
│   └── circuit_breaker()
├── ErrorClassifier
│   ├── classify()
│   └── is_transient()
└── CommRepair
    ├── recreate_communicator()
    └── reinitialize()
```

## Gray Areas

1. **Timeout threshold**: Fixed vs adaptive (based on network latency)
2. **Circuit breaker**: Global vs per-communicator state
3. **Error classification**: Heuristics vs explicit NCCL error codes
