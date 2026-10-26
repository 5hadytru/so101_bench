"""Comparable efficiency metrics over active policy control, excluding padded holds."""


def efficiency_metrics(control_wall_s, sim_loop_wall_s, control_sim_s, estimated_usd):
    if control_wall_s < 0 or sim_loop_wall_s < 0 or control_sim_s < 0:
        raise ValueError('Timing durations cannot be negative')
    # Simulation/rendering may run faster or slower than real time. Subtract its
    # measured wall duration, never its simulated seconds.
    frozen = max(0., control_wall_s - sim_loop_wall_s)
    return {
        'timing_revision': 'active-control-v1',
        'control_wall_s': control_wall_s,
        'sim_loop_wall_s': sim_loop_wall_s,
        'control_sim_s': control_sim_s,
        'estimated_usd': estimated_usd,
        'non_sim_wall_s': frozen,
        'non_sim_wall_per_control_sim_s': frozen/control_sim_s if control_sim_s else None,
        'estimated_usd_per_control_sim_s': estimated_usd/control_sim_s if control_sim_s else None,
        'definition': 'From agent startup to native terminal or policy stop, whichever first. '
                      'Exclude initial settling and subsequent scripted holding, finalization, '
                      'and simulator startup. Non-sim wall includes agent startup, model calls, '
                      'perception, IK, observation I/O and bridge overhead while physics is paused.',
    }


def aggregate_efficiency(records):
    """Use ratios of totals; absent historical timing stays unavailable."""
    measured = [r['efficiency'] for r in records if r.get('efficiency')]
    if len(measured) != len(records) or not measured:
        return None
    simulated = sum(r['control_sim_s'] for r in measured)
    paused = sum(r['non_sim_wall_s'] for r in measured)
    dollars = sum(r['estimated_usd'] for r in measured)
    return {'n': len(records), 'control_sim_s': simulated, 'non_sim_wall_s': paused,
            'estimated_usd': dollars,
            'non_sim_wall_per_control_sim_s': paused/simulated if simulated else None,
            'estimated_usd_per_control_sim_s': dollars/simulated if simulated else None}
