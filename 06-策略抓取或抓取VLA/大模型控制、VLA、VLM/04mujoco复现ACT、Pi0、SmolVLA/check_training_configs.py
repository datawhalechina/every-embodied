"""Parse tutorial YAMLs with the pinned LeRobot parser, without training."""

from pathlib import Path
import sys
from unittest.mock import patch

from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


@parser.wrap()
def check(cfg: TrainPipelineConfig):
    cfg.validate()
    assert isinstance(cfg.policy, (PI0Config, SmolVLAConfig))
    assert cfg.policy.n_action_steps <= cfg.policy.chunk_size
    assert cfg.optimizer is not None and cfg.scheduler is not None
    print(f"OK: {cfg.policy.type}, dataset={cfg.dataset.root}")
    return cfg


if __name__ == "__main__":
    for policy in ("pi0", "smolvla"):
        path = Path(__file__).resolve().parent / f"{policy}_datawhale_eai.yaml"
        # Match the documented two-token YAML argument, not JSON checkpoint loading.
        with patch.object(sys, "argv", [sys.argv[0], "--config_path", str(path)]):
            cfg = check()
        assert cfg.policy.type == policy
