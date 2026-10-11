"""Custom exceptions for CostRouter."""


class BudgetExhaustedError(RuntimeError):
    """Raised when the budget is exhausted and mode is HARD_STOP (REQ-BUDGET-1)."""

    def __init__(self, step_id: str) -> None:
        self.step_id = step_id
        super().__init__(
            f"Budget exhausted at step '{step_id}' — HARD_STOP mode is active."
        )


class MalformedOutputError(RuntimeError):
    """Raised when a model response fails the configured validator (REQ-ROUTER-6)."""

    def __init__(self, step_id: str) -> None:
        self.step_id = step_id
        super().__init__(
            f"Model output failed validation at step '{step_id}'."
        )
