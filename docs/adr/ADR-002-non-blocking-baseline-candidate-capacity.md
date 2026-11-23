# ADR-002 — Non-blocking Baseline Candidate Capacity

**Status:** Accepted  
**Date:** 2026-11-23  
**Scope:** FAR Local Audio Salience Detector  
**Implementation:** Task 13A; validation pending

## Context

FAR maintains bounded stores of learned baseline candidates. Existing resource-management policies protect retained identities, learned parameters, promotion blocks and transactional consistency.

During the Task 12B recorded-audio replay, candidate allocation exhausted the configured capacity of 128 records. The next allocation request caused the entire audio-processing transaction to be refused, despite otherwise valid input.

The diagnostic investigation established that:

- All 128 retained candidates were active and unready, associated with the same channel and four feature schemas.
- The nearest candidate's worst-band error was 61.4%, exceeding the configured 30% matching limit.
- The attempted 129th allocation was refused before an identity was assigned.
- 99 candidates had received only one analysis hop of learning.
- An alternate chunk partition reached the same failed analysis window.
- Transactional rollback preserved accepted coverage and learned-state consistency. No duplicate-ID accounting or rollback defect was identified.

The immediate architectural issue is that exhaustion of learning allocation capacity prevents sensory processing. Premature durable candidate creation is a separate policy issue.

## Decision

**Learning capacity exhaustion must not prevent sensory perception.**

Exhaustion of baseline candidate allocation capacity alone SHALL be treated as a learning limitation, not an audio admission failure.

### Sensory processing

When otherwise valid audio requires a new candidate but the candidate store is full, FAR SHALL continue audio admission, timeline advancement, feature extraction and salience assessment using available compatible models and evidence.

Assessments that cannot be supported by available baselines SHALL be explicitly represented as unavailable. Missing evidence SHALL NOT be represented as normality, zero salience or fabricated observations.

### Learning

FAR SHALL skip only the unavailable allocation and its dependent learning operations. Eligible updates to existing compatible models and candidates SHALL continue under existing learning policies.

A skipped allocation SHALL NOT create or consume a candidate identity. Unmatched observations SHALL NOT be forced into incompatible candidates.

Allocation may resume when capacity becomes available through existing authorized mechanisms and the applicable allocation eligibility conditions are satisfied.

### Observability

FAR SHALL report candidate allocation exhaustion explicitly through the existing SensorHealthEvent output family. The diagnostic SHALL distinguish a learning limitation from audio refusal or loss.

Repeated occurrences SHALL be aggregated according to existing health-event conventions. Recovery SHALL follow the existing diagnostic lifecycle conventions where applicable.

Skipped allocations SHALL NOT generate fabricated candidate lifecycle events. No additional output family is introduced.

### Transactional semantics

Candidate allocation exhaustion alone SHALL NOT roll back otherwise valid sensory processing.

Accepted audio coverage, temporal and feature state, valid learning updates, salience results and applicable diagnostics SHALL commit consistently under the existing transaction contract.

Actual admission failures, processing resource exhaustion and integrity violations remain subject to existing refusal and rollback policies. This decision does not authorize general exception suppression or partial commits after a failed transaction.

### Resource and identity invariants

- Configured candidate capacity SHALL NOT be exceeded.
- Retained identities, learned data, lifecycle states and promotion blocks SHALL be preserved, subject to otherwise authorized existing operations.
- No implicit eviction, deletion, replacement, merging or identity reuse is introduced.
- No promotion block may be bypassed.
- No unbounded pending allocation queue may be introduced.
- Existing matching, promotion, retention and learning eligibility policies remain unchanged.

## Alternatives considered

| Alternative | Disposition |
|---|---|
| Refuse the entire audio transaction | Rejected: couples perception availability to learning allocation capacity. |
| Increase candidate capacity | Rejected as a solution: postpones exhaustion without removing the dependency. |
| Automatically evict or replace candidates | Rejected: changes retention and identity guarantees and may discard learned evidence. |
| Disable all learning when capacity is exhausted | Rejected: unnecessarily prevents valid updates to existing models. |
| Continue sensory processing and skip unavailable allocations | Accepted: preserves bounded resources while maintaining perception where supported by available evidence. |

## Consequences

### Benefits

- Sensory processing remains available during candidate allocation saturation.
- Existing learned evidence and identity guarantees are preserved.
- Learning degradation becomes observable without falsely reporting audio loss.

### Trade-offs

- Novel patterns may remain unlearned while allocation capacity is exhausted.
- Baseline-dependent assessments may be unavailable or limited by existing models.
- Candidate saturation may persist until capacity is released through authorized mechanisms.
- Diagnostic aggregation and transactional consistency require explicit regression coverage.

## Validation requirements

Task 13A SHALL verify successful admission and continued supported salience detection at full candidate capacity; eligible existing-model learning; explicit aggregated diagnostics; preservation of identities and promotion blocks; allocation recovery; and unchanged rollback behavior for unrelated failures.

The original Task 12B failure and alternate partition SHALL be replayed to verify that the previously refused analysis window can be processed without exceeding candidate capacity. Partition behavior SHALL satisfy the existing equivalence contract.

Run the unit suite and document the results. Frozen positive-pilot scoring is a subsequent task, not evidence of implementation correctness for this ADR.

## Out of scope

Candidate-specific stability evidence, shared eligibility timer redesign, candidate matching thresholds, retention and promotion policy changes, hardware timing, and broader acoustic accuracy validation are excluded.

The premature creation of durable candidate identities remains an open architectural issue for Task 13B and requires a separate decision.

## Related documents

- DESIGN-001 and DESIGN-002 — existing streaming, learning and resource-management contracts.
- DESIGN-003 — acoustic salience model.
- ADR-001 — existing storage representation decision.
- Task 12B capacity diagnostic — motivating evidence.
- Task 13A — implementation and validation.
