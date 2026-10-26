"""Distinguish stock Inspect policy exhaustion from integration/provider errors."""


def classify_exception(exc):
    # These are the stock agent's explicit failure messages. Do not classify
    # arbitrary RuntimeErrors (including provider failures) as model outcomes.
    if isinstance(exc, RuntimeError) and str(exc).startswith((
        "LLM tool calls kept failing; last error:",
        "LLM produced no tool call in ",
    )):
        return "policy_error", None
    return "infrastructure_error", type(exc).__name__
