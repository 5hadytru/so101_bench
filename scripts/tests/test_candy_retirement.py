"""Retirement preserves stable IDs and non-grasp scenes while excluding grasps."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('retirement_benchmark', ROOT/'source/so101_bench/so101_bench/benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)
sys.path.insert(0, str(ROOT/'scripts'))
from primary_eval_exclusions import EXCLUDED_TRIAL_IDS, primary_episode_excluded


def test_retired_candy_is_not_regenerated_even_with_private_catalog():
    assert 'candy bar' not in benchmark.BENCHMARK_OBJECT_NAMES
    assert all('candy bar' not in objects for objects in benchmark.OBJECT_SPLITS.values())
    assert benchmark.object_metadata('candy bar')  # historical props still resolve


def test_stable_primary_ids_and_all_six_grasp_cases_are_excluded():
    episodes = benchmark.load_episode_jsonl(ROOT/'tasks/primary_eval_v2.jsonl')
    assert len(episodes) == 924
    assert [i for i,e in enumerate(episodes) if not benchmark.episode_is_executable(e)] == [615,681,694,712,787,862]
    assert sum(not primary_episode_excluded(e) for e in episodes) == 916
    for i in (615,681,694,712,787,862):
        with pytest.raises(ValueError, match='retired object'):
            episodes[i].reset_payload()


def test_non_grasp_candy_scenes_remain_executable_and_other_dataset_ids_do_not_collide():
    episodes = benchmark.load_episode_jsonl(ROOT/'tasks/primary_eval_v2.jsonl')
    props = [e for e in episodes if 'candy bar' in e.objects and benchmark.episode_is_executable(e)]
    assert len(props) == 22
    assert all(e.reset_payload() for e in props)
    assert all(not primary_episode_excluded(e) for e in props)
    # An unrelated task carrying the same numeric ID must not be discarded.
    from dataclasses import replace
    different = replace(episodes[0], metadata={'trial_id':615})
    assert not primary_episode_excluded(different)
