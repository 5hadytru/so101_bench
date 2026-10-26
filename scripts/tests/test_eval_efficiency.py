"""Avoid real-time assumptions and denominator artifacts in the appendix table."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from eval_efficiency import efficiency_metrics, aggregate_efficiency


def test_real_simulation_wall_time_not_simulated_seconds_is_subtracted():
    # One simulated second took two wall seconds; agent computation took five.
    fast = efficiency_metrics(7.,2.,1.,.01)
    assert fast['non_sim_wall_s']==5.
    assert fast['non_sim_wall_per_control_sim_s']==5.
    idle = efficiency_metrics(5.,0.,0.,.01)
    assert idle['non_sim_wall_per_control_sim_s'] is None
    assert idle['estimated_usd_per_control_sim_s'] is None


def test_global_metrics_are_ratios_of_totals_and_missing_historical_time_is_not_zero():
    a={'efficiency':efficiency_metrics(7.,2.,1.,.01)}
    b={'efficiency':efficiency_metrics(9.,5.,10.,.02)}
    overall=aggregate_efficiency([a,b])
    assert overall['non_sim_wall_per_control_sim_s']==pytest.approx(9/11)
    assert overall['estimated_usd_per_control_sim_s']==pytest.approx(.03/11)
    assert aggregate_efficiency([a,{}]) is None


def test_native_step_timing_includes_rendering_and_survives_a_failed_step():
    source=ROOT/'source/so101_bench/so101_bench/utils/llm_harness.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SimArm')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    clock=[0.]
    ns={'time':SimpleNamespace(monotonic=lambda:clock[0])}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
    def native():
        clock[0]+=2.
        return True
    arm=SimpleNamespace(sim_loop_wall_s=0.,_step_native=native)
    assert ns['step'](arm) is True
    assert arm.sim_loop_wall_s==2.
    def broken():
        clock[0]+=3.
        raise ValueError('step failure')
    arm._step_native=broken
    with pytest.raises(ValueError):ns['step'](arm)
    assert arm.sim_loop_wall_s==5.
