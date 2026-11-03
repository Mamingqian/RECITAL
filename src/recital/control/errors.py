"""Control-layer failures.

None of these derive from ValueError, KeyError or TypeError. A control failure
stops the cell; it is not fed back to the agent as a revisable mistake.
"""


class ControlError(Exception):
    """Base class of every failure the control layer raises on purpose."""


class PromptBudgetExceeded(ControlError):
    """A rendered prompt is larger than the declared per-call token cap."""


class BlindingViolation(ControlError):
    """An outgoing message or tool result names something the arm must not see."""


class RunTokenBudgetExceeded(ControlError):
    """The run's cumulative token usage reached the declared cap."""


class LedgerCorruption(ControlError):
    """The event log, a blob, or the event sequence is internally inconsistent."""


class JobInfraError(ControlError):
    """A training job failed for a reason that belongs to the cluster, not the recipe."""


class SchedulerUnreachable(JobInfraError):
    """The scheduler did not answer, so nothing is known about the job. Ask again later."""


class ScoringFailed(ControlError):
    """A scorer failed on a pool. The pool and the training behind it are on disk, so the job can be run again."""
