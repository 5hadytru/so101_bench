"""SO-101 Bench for LeRobot: importing this package registers ``--env.type=so101_bench``.

LeRobot imports installed ``lerobot_env_*`` packages automatically (third-party plugin discovery).
LeRobot versions that ship SO-101 Bench themselves already register the type; then this package
only provides the simulator server and the scorer.
"""

from lerobot.envs.configs import EnvConfig

if "so101_bench" not in EnvConfig.get_known_choices():
    from .config import SO101BenchEnvConfig

    __all__ = ["SO101BenchEnvConfig"]
