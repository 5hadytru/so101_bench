"""Require native camera/physics evidence even when Kit masks a Python error."""
import json
from pathlib import Path


def validate(folder):
    folder = Path(folder)
    proof = json.loads((folder/'smoke.json').read_text())
    assert proof.get('ok') is True, 'Native smoke did not pass'
    assert proof.get('native_steps', 0) > 45, 'No native action after settling'
    shapes = proof.get('rgb_shapes', {})
    assert len(shapes) == 2, 'Both native cameras required'
    assert all(len(s) == 3 and min(s[:2]) > 0 and s[2] in (3, 4)
               for s in shapes.values()), 'Invalid native RGB shapes'
    assert (folder/'rollout_overhead_wrist.mp4').stat().st_size > 0, 'Missing camera video'
    return proof


if __name__ == '__main__':
    import sys
    print(json.dumps(validate(sys.argv[1])))
