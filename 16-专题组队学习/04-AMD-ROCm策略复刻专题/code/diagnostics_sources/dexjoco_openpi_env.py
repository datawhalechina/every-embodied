"""DexJoCo environment wrapper for OpenPI inference."""

import copy

import numpy as np
from dexjoco.tasks import CONFIG_MAPPING
from openpi_client import image_tools
from scipy.spatial.transform import Rotation as R
from typing import Literal


class DexJoCoOpenPIEnv:
    """Adapt a DexJoCo simulation environment to the OpenPI client interface.

    The wrapper exposes processed image observations, robot state, and prompt
    fields in the format expected by the OpenPI policy server. It also converts
    policy actions from rotation-vector format to the quaternion-based action
    format used by the underlying DexJoCo environment.
    """

    def __init__(
        self,
        env_name: str,
        camera_mapping: dict[str, str],
        seed: int,
        rand_full: bool,
        randomize_dynamics: bool,
        dual_arm: bool,
        prompt: str,
        render_mode: Literal["rgb_array", "human"],
        pad_state_dim46: bool = False,
        password: list[int] | None = None,
    ):
        """Create a wrapper for one DexJoCo task environment.

        Args:
            env_name: Name used to look up the DexJoCo task config.
            camera_mapping: Mapping from OpenPI observation keys to DexJoCo
                camera keys, for example {"base": "front"}.
            seed: Environment seed.
            rand_full: Whether to randomize the environment scene.
            randomize_dynamics: Whether to randomize environment dynamics.
            dual_arm: Whether the task uses the dual-arm state/action layout.
            prompt: Language prompt passed to the policy.
            render_mode: Render mode forwarded to the DexJoCo environment.
            pad_state_dim46: Whether to pad single-arm state to 46 dimensions.
            password: List of digits representing the password for the iPad unlock task.
        """
        self.env_name = env_name
        self.camera_mapping = camera_mapping
        self.seed = seed
        self.rand_full = rand_full
        self.randomize_dynamics = randomize_dynamics
        self.dual_arm = dual_arm
        self.prompt = prompt
        self.render_mode: Literal["rgb_array", "human"] = render_mode
        self.pad_state_dim46 = pad_state_dim46
        self.password = password

        self.env = None
        # Processed one-frame observation used as OpenPI policy input.
        self.obs = {}
        # Latest raw camera frames, kept at original resolution for recording.
        self._raw_obs: dict = {}
        self._done = False
        self._success = False

    def start(self):
        """Instantiate the underlying DexJoCo simulation environment."""
        config = CONFIG_MAPPING[self.env_name]()

        if self.env_name == "bimanual_unlock_ipad" and self.password is not None:
            env_kwargs = {"password": self.password}
        else:
            env_kwargs = {}
            if self.password is not None:
                print(
                    f"Warning: password provided but will be ignored for env {self.env_name}"
                )

        self.env = config.get_environment(
            policy_mode=True,
            render_mode=self.render_mode,
            randomize=self.rand_full,
            seed=self.seed,
            randomize_dynamics=self.randomize_dynamics,
            **env_kwargs,
        )

    def close(self):
        """Close the underlying simulation environment."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def reset(self):
        """Reset the environment and refresh the current processed observation."""
        assert self.env is not None, "Environment not started. Call start() first."
        obs, _ = self.env.reset()
        self._done = False
        self._success = False

        self._update_raw_obs(obs)
        self.obs = self._process_obs(obs)

    def step(self, action: np.ndarray):
        """Execute one OpenPI-format action in the DexJoCo environment.

        Args:
            action: Policy action in rotation-vector format. Single-arm actions
                use [xyz(3), rotvec(3), hand(16)]. Dual-arm actions use
                [r_xyz(3), r_rotvec(3), r_hand(16),
                l_xyz(3), l_rotvec(3), l_hand(16)].
        """
        assert self.env is not None, "Environment not started. Call start() first."
        env_action = self._process_action(action)
        obs, reward, terminated, truncated, info = self.env.step(env_action)

        self._done = bool(terminated)
        self._success = info.get("succeed", False)

        self._update_raw_obs(obs)
        self.obs = self._process_obs(obs)

        return info.get("pressed_digits")

    def get_obs(self) -> dict[str, np.ndarray]:
        """Return a copy of the latest processed policy observation."""
        return copy.deepcopy(self.obs)

    def stay(self, continue_stay: bool = False):
        """Hold the current robot pose by sending the current state as an action."""
        if continue_stay:
            stay_state = self.last_stay_state
        else:
            stay_state = self.obs["state"]
            self.last_stay_state = stay_state

        if self.dual_arm:
            # state: [r_arm(7), l_arm(7), r_hand(16), l_hand(16)]
            # action:
            # [r_xyz(3), r_rotvec(3), r_hand(16), l_xyz(3), l_rotvec(3), l_hand(16)]
            r_arm = stay_state[:7]
            l_arm = stay_state[7:14]
            r_hand = stay_state[14:30]
            l_hand = stay_state[30:46]

            r_xyz = r_arm[:3]
            r_quat = r_arm[3:7]
            r_rotvec = R.from_quat(r_quat, scalar_first=True).as_rotvec()

            l_xyz = l_arm[:3]
            l_quat = l_arm[3:7]
            l_rotvec = R.from_quat(l_quat, scalar_first=True).as_rotvec()

            action = np.concatenate([r_xyz, r_rotvec, r_hand, l_xyz, l_rotvec, l_hand])
        else:
            # state: [arm(7), hand(16)]
            # action: [xyz(3), rotvec(3), hand(16)]
            arm = stay_state[:7]
            hand = stay_state[7:23]

            xyz = arm[:3]
            quat = arm[3:7]
            rotvec = R.from_quat(quat, scalar_first=True).as_rotvec()

            action = np.concatenate([xyz, rotvec, hand])

        return self.step(action)

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_success(self) -> bool:
        return self._success

    def _process_obs(self, env_obs: dict) -> dict[str, np.ndarray]:
        """Convert DexJoCo observations to OpenPI policy input format.

        Images are resized to 224x224 and converted to uint8. State values stay
        in the environment coordinate system and are not normalized here.
        """
        obs_dict = {}

        for policy_key, env_key in self.camera_mapping.items():
            img = env_obs[env_key]
            obs_dict[policy_key] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, 224, 224)
            )

        if self.dual_arm:
            state = env_obs["state"][:46]
        else:
            state = env_obs["state"][:23]
            if self.pad_state_dim46:
                state = np.concatenate([state, np.zeros(46 - len(state))])
        obs_dict["state"] = state
        obs_dict["prompt"] = self.prompt

        return obs_dict

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Convert OpenPI policy actions to DexJoCo environment action format."""
        if self.dual_arm:
            r_xyz = action[:3]
            r_rotvec = action[3:6]
            r_hand = action[6:22]
            l_xyz = action[22:25]
            l_rotvec = action[25:28]
            l_hand = action[28:44]

            r_quat = R.from_rotvec(r_rotvec).as_quat(scalar_first=True)
            l_quat = R.from_rotvec(l_rotvec).as_quat(scalar_first=True)

            return np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand])
        else:
            xyz = action[:3]
            rotvec = action[3:6]
            hand = action[6:22]

            quat = R.from_rotvec(rotvec).as_quat(scalar_first=True)

            return np.concatenate([xyz, quat, hand])

    def _update_raw_obs(self, env_obs: dict):
        """Cache raw camera frames from the latest environment observation."""
        self._raw_obs = {}
        for env_key in self.camera_mapping.values():
            self._raw_obs[env_key] = env_obs[env_key]

    def get_raw_images(self) -> dict[str, np.ndarray]:
        """Return latest raw camera frames at their original resolution."""
        return self._raw_obs
