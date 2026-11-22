"""Administrative baseline lifecycle only. No learned acoustic parameters."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace

from .contracts import (
    BaselineCompatibility,
    BaselineRecord,
    CandidateCommand,
    CommandReceipt,
    LifecycleKind,
    StructuralInputError,
)


@dataclass(frozen=True, slots=True)
class ReadinessEvidence:
    """Explicit extension decision; FAR does not calculate readiness metrics."""

    candidate_id: str
    evidence_revision: int
    ready: bool
    materially_changed: bool
    compatibility: tuple[BaselineCompatibility, ...]
    explicit_review: bool = False


@dataclass(frozen=True, slots=True)
class ControlPlan:
    command: CandidateCommand
    before: BaselineRecord | None
    after: BaselineRecord | None
    revision: int
    blocked: bool
    event_type: LifecycleKind | None
    code: str | None = None
    prior: CommandReceipt | None = None


class BaselineStore:
    """Bounded metadata and version guards, independent of signal processing."""

    def __init__(self, max_receipts: int) -> None:
        self.records: dict[str, BaselineRecord] = {}
        self.administration: dict[str, tuple[int, bool]] = {}
        self.receipts: OrderedDict[str, tuple[CandidateCommand, CommandReceipt]] = (
            OrderedDict()
        )
        self.max_receipts = max_receipts

    def plan(self, command: CandidateCommand) -> ControlPlan:
        if not isinstance(command, CandidateCommand):
            raise StructuralInputError("expected CandidateCommand")
        if (
            not command.command_id
            or not command.candidate_id
            or type(command.expected_control_revision) is not int
            or command.expected_control_revision < 0
        ):
            raise StructuralInputError("invalid command identity/revision")
        if command.action not in (
            "approve",
            "reject",
            "block",
            "unblock",
            "archive",
            "reactivate",
        ):
            raise StructuralInputError("unknown candidate action")
        before = self.records.get(command.candidate_id)
        revision, blocked = self.administration.get(command.candidate_id, (0, False))
        plan = ControlPlan(command, before, before, revision, blocked, None)
        old = self.receipts.get(command.command_id)
        if old is not None:
            if old[0] != command:
                return replace(plan, code="command_id_conflict")
            return replace(plan, prior=old[1])
        if command.candidate_id not in self.administration:
            return replace(plan, code="unknown_candidate")
        if command.expected_control_revision != revision:
            return replace(plan, code="stale_control_revision")
        if before is None and command.action not in ("block", "unblock"):
            return replace(plan, code="learning_was_reset")
        if command.action in ("approve", "reject"):
            assert before is not None
            if before.role != "candidate":
                return replace(plan, code="not_candidate")
            if command.proposal_id is None or command.proposal_id != before.proposal_id:
                return replace(plan, code="stale_proposal")
        event: LifecycleKind | None
        after = before
        if command.action == "block":
            blocked, event = True, "promotion_blocked" if not blocked else None
        elif command.action == "unblock":
            blocked, event = False, "promotion_unblocked" if blocked else None
        elif command.action == "reject":
            assert before is not None
            after = replace(
                before, ready=False, rejected_proposal_id=before.proposal_id
            )
            event = "promotion_rejected"
        elif command.action == "approve":
            assert before is not None
            if blocked:
                return replace(plan, code="promotion_blocked")
            if not before.ready or before.activity != "active":
                return replace(plan, code="not_ready")
            if not before.compatibility or any(
                c.status != "compatible" for c in before.compatibility
            ):
                return replace(plan, code="incompatible")
            after = replace(before, role="established", ready=False)
            event = "candidate_promoted"
        elif command.action == "archive":
            assert before is not None
            event = None if before.activity == "archived" else "baseline_archived"
            after = replace(before, activity="archived", ready=False)
        else:
            assert before is not None
            if before.activity != "archived":
                return replace(plan, code="not_archived")
            after = replace(
                before,
                activity="reactivating",
                ready=False,
                compatibility=tuple(
                    replace(c, status="revalidate") for c in before.compatibility
                ),
            )
            event = "baseline_reactivation_started"
        # Even no-op new commands advance the guard, so expired receipts are safe.
        if revision >= 2**63 - 1:
            return replace(plan, code="control_revision_exhausted")
        revision += 1
        if after is not None:
            after = replace(after, control_revision=revision, promotion_blocked=blocked)
        return replace(
            plan, after=after, revision=revision, blocked=blocked, event_type=event
        )

    def commit(self, plan: ControlPlan, event_ids: tuple[str, ...]) -> CommandReceipt:
        if plan.prior is not None:
            return plan.prior
        if plan.code is None:
            self.administration[plan.command.candidate_id] = (
                plan.revision,
                plan.blocked,
            )
            if plan.after is not None:
                self.records[plan.command.candidate_id] = plan.after
        receipt = CommandReceipt(
            plan.command.command_id,
            plan.command.candidate_id,
            "refused" if plan.code else "applied" if plan.event_type else "no_change",
            plan.revision,
            event_ids,
        )
        # A conflicting ID must never overwrite the original receipt.
        if plan.code != "command_id_conflict":
            self.receipts[plan.command.command_id] = (plan.command, receipt)
            while len(self.receipts) > self.max_receipts:
                self.receipts.popitem(last=False)
        return receipt
