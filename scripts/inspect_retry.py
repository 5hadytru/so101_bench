"""Continue a stock Inspect conversation after recoverable tool rejections."""
from inspect_outcomes import classify_exception


def act_with_recovery(policy,observe,max_retries,on_retry):
    for attempt in range(max_retries+1):
        try:
            return policy.act(observe())
        except RuntimeError as exc:
            stop,error=classify_exception(exc)
            if stop!='policy_error' or error is not None or attempt>=max_retries:
                raise
            # Preserve policy, call counter, transcript and measured robot state.
            on_retry(str(exc))
