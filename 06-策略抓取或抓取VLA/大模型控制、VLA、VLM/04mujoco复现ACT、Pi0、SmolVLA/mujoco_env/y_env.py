import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw
import re
from pathlib import Path

class SimpleEnv:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                seed = None):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        # Arm joint names default (legacy OMY setting).
        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
        ]
        # Auto-detect robot bindings for different robots (e.g., OMY / Nova5).
        self._configure_robot_bindings()
        self.init_viewer()
        self.reset(seed)
        # Deadband for teleop action to prevent IK jitter around zero input.
        self.action_deadband = 1e-6
        # Smooth gripper command to avoid instant jump.
        # This is command slew-rate (software-side), not actuator torque.
        # xarm7 uses tendon gripper; keep a lower default to reduce contact penetration.
        default_gripper_rate = "0.015" if self.ee_body_name == 'xarm_gripper_base_link' else "0.03"
        self.gripper_rate_per_step = float(os.getenv("GRIPPER_RATE_PER_STEP", default_gripper_rate))
        # Success-rule params (tunable without touching notebook logic).
        self.success_xy_threshold = float(os.getenv("SUCCESS_XY_THRESHOLD", "0.10"))
        self.success_z_threshold = float(os.getenv("SUCCESS_Z_THRESHOLD", "0.12"))
        self.success_ee_height_threshold = float(
            os.getenv("SUCCESS_EE_HEIGHT_THRESHOLD", "0.95" if self.ee_body_name == 'Link6' else "0.90")
        )
        self.success_require_gripper_open = os.getenv("SUCCESS_REQUIRE_GRIPPER_OPEN", "1") != "0"
        # gripper open score in [0,1], where 1=open, 0=closed
        self.success_gripper_open_norm_threshold = float(
            os.getenv("SUCCESS_GRIPPER_OPEN_NORM_THRESHOLD", "0.30")
        )
        # xarm7: stop further closing when fingers already contact mug (anti-penetration guard).
        self.xarm7_contact_stop_close = os.getenv("XARM7_CONTACT_STOP_CLOSE", "1") != "0"
        # IK backend: `pybullet` or `mujoco`
        default_ik_backend = 'mujoco'
        if self.ee_body_name in ('link6', 'xarm_gripper_base_link'):
            default_ik_backend = 'pybullet'
        self.ik_backend = os.getenv("IK_BACKEND", default_ik_backend).strip().lower()
        self.pb = None
        self.pb_client = None
        self.pb_robot = None
        self.pb_joint_name_to_index = {}
        self.pb_dof_joint_indices = []
        self.pb_joint_sol_index = {}
        self.pb_ee_link_index = None
        # Alignment from MuJoCo EE frame/world to PyBullet EE frame/world.
        self.pb_pos_offset = None
        self.pb_R_mj_to_pb = None
        self._init_pybullet_ik()
        # Teleop / IK tuning knobs.
        is_xarm_like = self.ee_body_name in ('link6', 'xarm_gripper_base_link')
        self.teleop_pos_step = float(os.getenv("TELEOP_POS_STEP", "0.004" if is_xarm_like else "0.007"))
        self.teleop_rot_step = float(os.getenv("TELEOP_ROT_STEP", str(0.1 * 0.3)))
        self.ik_max_tick = int(os.getenv("IK_MAX_TICK", "120" if is_xarm_like else "50"))
        self.ik_stepsize = float(os.getenv("IK_STEPSIZE", "1.0"))
        self.ik_eps = float(os.getenv("IK_EPS", "5e-3" if is_xarm_like else "1e-2"))
        self.ik_trim_th = float(os.getenv("IK_TRIM_TH_DEG", "2.0" if is_xarm_like else "5.0")) * np.pi / 180.0
        # Per-step joint delta clamp to avoid IK jumps / singular flips.
        self.ik_max_joint_delta = float(
            os.getenv(
                "IK_MAX_JOINT_DELTA",
                "0.06" if self.ee_body_name == 'xarm_gripper_base_link' else "0.12"
            )
        )
        self.ik_pos_err_max = float(
            os.getenv(
                "IK_POS_ERR_MAX",
                "0.03" if self.ee_body_name == 'xarm_gripper_base_link' else "0.05"
            )
        )
        # Null-space rest-pose bias for redundant arms (mainly xarm7).
        self.ik_rest_bias = float(
            os.getenv(
                "IK_REST_BIAS",
                "0.18" if self.ee_body_name == 'xarm_gripper_base_link' else "0.0"
            )
        )
        self.ik_rest_pose = None

    def _init_pybullet_ik(self):
        """Initialize a lightweight PyBullet URDF model for IK (xarm6/xarm7)."""
        if self.ik_backend != 'pybullet':
            return
        try:
            import pybullet as pb  # type: ignore
        except Exception:
            print("[IK] pybullet is not installed. Falling back to mujoco IK.")
            self.ik_backend = 'mujoco'
            return

        if self.ee_body_name == 'xarm_gripper_base_link':
            robot_urdf_rel = './asset/pybullet_urdf/xarm7_gripper.urdf'
        elif self.ee_body_name == 'link6':
            robot_urdf_rel = './asset/pybullet_urdf/xarm6_gripper.urdf'
        else:
            print(f"[IK] No pybullet model mapping for ee body `{self.ee_body_name}`. Falling back to mujoco IK.")
            self.ik_backend = 'mujoco'
            return

        try:
            self.pb = pb
            self.pb_client = pb.connect(pb.DIRECT)
            pb.resetSimulation(physicsClientId=self.pb_client)
            pb.setGravity(0, 0, 0, physicsClientId=self.pb_client)
            # Prefer ASCII junction path to avoid Windows unicode path issues in pybullet.
            ascii_root = Path.home() / "lerobot-mujoco-tutorial"
            project_root = ascii_root if ascii_root.exists() else Path.cwd()
            rel_norm = robot_urdf_rel.replace('\\', '/').lstrip('./')
            full_urdf = str((project_root / rel_norm)).replace('\\', '/')
            if not os.path.exists(full_urdf):
                raise FileNotFoundError(full_urdf)
            self.pb_robot = int(
                pb.loadURDF(
                    full_urdf,
                    basePosition=[0.0, 0.0, 0.0],
                    baseOrientation=[0.0, 0.0, 0.0, 1.0],
                    useFixedBase=True,
                    flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                    physicsClientId=self.pb_client,
                )
            )

            for j in range(pb.getNumJoints(self.pb_robot, physicsClientId=self.pb_client)):
                info = pb.getJointInfo(self.pb_robot, j, physicsClientId=self.pb_client)
                jname = info[1].decode('utf-8')
                lname = info[12].decode('utf-8')
                jtype = int(info[2])
                if len(jname) > 0:
                    self.pb_joint_name_to_index[jname] = j
                if jtype != pb.JOINT_FIXED:
                    self.pb_joint_sol_index[j] = len(self.pb_dof_joint_indices)
                    self.pb_dof_joint_indices.append(j)
                if lname == self.ee_body_name:
                    self.pb_ee_link_index = j

            if self.pb_ee_link_index is None:
                raise RuntimeError(f"PyBullet link `{self.ee_body_name}` not found")
            print(f"[IK] Using pybullet IK backend. ee_link={self.ee_body_name} idx={self.pb_ee_link_index}")
        except Exception as e:
            print(f"[IK] Failed to initialize pybullet IK ({e}). Falling back to mujoco IK.")
            self.ik_backend = 'mujoco'
            self.pb = None
            self.pb_client = None
            self.pb_robot = None
            self.pb_joint_name_to_index = {}
            self.pb_dof_joint_indices = []
            self.pb_joint_sol_index = {}
            self.pb_ee_link_index = None

    def _solve_ik_pybullet(self, q_seed, p_trgt, R_trgt):
        """Solve arm IK with PyBullet and return q for self.joint_names."""
        pb = self.pb
        cid = self.pb_client
        rid = self.pb_robot
        if pb is None or cid is None or rid is None or self.pb_ee_link_index is None:
            raise RuntimeError("PyBullet IK backend not ready")

        # Sync current seed state to keep IK branch stable.
        for i, jn in enumerate(self.joint_names):
            if jn in self.pb_joint_name_to_index:
                pb.resetJointState(
                    rid,
                    self.pb_joint_name_to_index[jn],
                    float(q_seed[i]),
                    targetVelocity=0.0,
                    physicsClientId=cid,
                )

        # Calibrate static world/frame alignment between MuJoCo and PyBullet EE.
        ls_curr = pb.getLinkState(
            rid, self.pb_ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=cid
        )
        p_pb_curr = np.array(ls_curr[4], dtype=np.float32)
        q_pb_curr = np.array(ls_curr[5], dtype=np.float32)
        R_pb_curr = np.array(pb.getMatrixFromQuaternion(q_pb_curr.tolist()), dtype=np.float32).reshape(3, 3)
        p_mj_curr, R_mj_curr = self.env.get_pR_body(body_name=self.ee_body_name)
        p_mj_curr = np.asarray(p_mj_curr, dtype=np.float32)
        R_mj_curr = np.asarray(R_mj_curr, dtype=np.float32)
        self.pb_pos_offset = p_pb_curr - p_mj_curr
        self.pb_R_mj_to_pb = R_mj_curr.T @ R_pb_curr

        p_pb_trgt = np.asarray(p_trgt, dtype=np.float32) + self.pb_pos_offset
        R_pb_trgt = np.asarray(R_trgt, dtype=np.float32) @ self.pb_R_mj_to_pb
        quat = pb.getQuaternionFromEuler(r2rpy(R_pb_trgt).tolist())

        # Null-space parameters for redundant xarm7 IK.
        lower_limits, upper_limits, joint_ranges, rest_poses, joint_damping = [], [], [], [], []
        q_seed_by_name = {jn: float(q_seed[i]) for i, jn in enumerate(self.joint_names)}
        q_rest_by_name = {}
        if self.ik_rest_pose is not None:
            for i, jn in enumerate(self.joint_names):
                if i < len(self.ik_rest_pose):
                    q_rest_by_name[jn] = float(self.ik_rest_pose[i])
        for j in self.pb_dof_joint_indices:
            info = pb.getJointInfo(rid, j, physicsClientId=cid)
            lo = float(info[8])
            hi = float(info[9])
            if hi < lo:
                lo, hi = hi, lo
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = -2.0 * np.pi, 2.0 * np.pi
            lower_limits.append(lo)
            upper_limits.append(hi)
            joint_ranges.append(max(hi - lo, 1e-3))
            jn = info[1].decode('utf-8')
            if jn in q_seed_by_name:
                q_seed_j = q_seed_by_name[jn]
                q_rest_j = q_rest_by_name.get(jn, q_seed_j)
                q_pref = (1.0 - self.ik_rest_bias) * q_seed_j + self.ik_rest_bias * q_rest_j
                rest_poses.append(float(q_pref))
                joint_damping.append(0.02)
            else:
                rest_poses.append(float(pb.getJointState(rid, j, physicsClientId=cid)[0]))
                joint_damping.append(0.10)

        sol = pb.calculateInverseKinematics(
            rid,
            self.pb_ee_link_index,
            targetPosition=p_pb_trgt.tolist(),
            targetOrientation=quat,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            jointDamping=joint_damping,
            maxNumIterations=max(self.ik_max_tick, 64),
            residualThreshold=1e-4,
            physicsClientId=cid,
        )

        q_out = []
        for i, jn in enumerate(self.joint_names):
            if jn in self.pb_joint_name_to_index:
                jidx = self.pb_joint_name_to_index[jn]
                # `sol` is indexed by movable joint order, not raw joint index.
                sidx = self.pb_joint_sol_index.get(jidx, None)
                if (sidx is not None) and (sidx < len(sol)):
                    q_out.append(float(sol[sidx]))
                else:
                    q_out.append(float(q_seed[i]))
            else:
                q_out.append(float(q_seed[i]))
        q_out = np.array(q_out, dtype=np.float32)

        # Clamp per-step update to avoid sudden configuration flips.
        dq = q_out - np.asarray(q_seed, dtype=np.float32)
        dq = np.clip(dq, -self.ik_max_joint_delta, self.ik_max_joint_delta)
        q_safe = np.asarray(q_seed, dtype=np.float32) + dq

        # Validate solved EE position; reject pathological solutions.
        for i, jn in enumerate(self.joint_names):
            jidx = self.pb_joint_name_to_index.get(jn, None)
            if jidx is not None:
                pb.resetJointState(
                    rid, jidx, float(q_safe[i]), targetVelocity=0.0, physicsClientId=cid
                )
        ls_chk = pb.getLinkState(
            rid, self.pb_ee_link_index, computeForwardKinematics=True, physicsClientId=cid
        )
        p_pb_chk = np.array(ls_chk[4], dtype=np.float32)
        pos_err = float(np.linalg.norm(p_pb_chk - p_pb_trgt))
        if pos_err > self.ik_pos_err_max:
            return np.asarray(q_seed, dtype=np.float32)
        return q_safe

    def _safe_names(self, names):
        return [n for n in names if isinstance(n, str) and len(n) > 0]

    def _pick_first_available(self, candidates, name_pool):
        for c in candidates:
            if c in name_pool:
                return c
        return None

    def _configure_robot_bindings(self):
        """Auto-configure end-effector/gripper/camera names for different robot XMLs."""
        body_names = set(self._safe_names(getattr(self.env, 'body_names', [])))
        joint_names_all = self._safe_names(getattr(self.env, 'joint_names', []))
        rev_joint_names = self._safe_names(getattr(self.env, 'rev_joint_names', []))
        cam_names = set(self._safe_names(getattr(self.env, 'cam_names', [])))

        # Prefer revolute joint list if available.
        if len(rev_joint_names) > 0:
            arm = [j for j in rev_joint_names if re.match(r'^joint\d+$', str(j))]
            arm = sorted(arm, key=lambda x: int(x.replace('joint', '')))
            if len(arm) >= 6:
                # Keep all continuous arm joints (xArm7 has joint1~joint7).
                self.joint_names = arm
            elif len(arm) > 0:
                self.joint_names = arm
            self.gripper_joint_names = [j for j in rev_joint_names if j not in self.joint_names]
        else:
            self.gripper_joint_names = [j for j in joint_names_all if j not in self.joint_names]

        # Nova5 (Robotiq) should be driven by a single master finger joint.
        # Other finger joints are coupled by MJCF equality mimic constraints.
        if ('finger_joint' in self.gripper_joint_names) and ('right_outer_knuckle_joint' in joint_names_all):
            self.gripper_joint_names = ['finger_joint']
        # xArm7 menagerie: only drive driver joints for gripper.
        if ('left_driver_joint' in joint_names_all) and ('right_driver_joint' in joint_names_all):
            self.gripper_joint_names = ['left_driver_joint', 'right_driver_joint']
        # UFactory lite6 gripper: only left finger has actuator, right finger is mimic/equality.
        if ('gripper_left_finger' in joint_names_all) and ('gripper_right_finger' in joint_names_all):
            self.gripper_joint_names = ['gripper_left_finger']
        self.control_joint_names = list(self.joint_names) + list(self.gripper_joint_names)

        # End-effector body for IK / visualization / success checking.
        self.ee_body_name = self._pick_first_available(
            [
                'tcp_link',
                'xarm_gripper_base_link',
                'link6',
                'link_tcp',
                'hand',
                'Link6',
                'right_outer_knuckle',
                'left_outer_knuckle',
            ],
            body_names,
        )
        if self.ee_body_name is None:
            self.ee_body_name = self._safe_names(getattr(self.env, 'body_names', []))[-1]

        # Gripper monitor joint for binary state & success criterion.
        self.gripper_monitor_joint = self._pick_first_available(
            ['rh_r1', 'finger_joint', 'left_inner_finger_joint', 'right_inner_finger_joint'],
            set(joint_names_all),
        )
        if self.gripper_monitor_joint is None and len(self.gripper_joint_names) > 0:
            self.gripper_monitor_joint = self.gripper_joint_names[0]
        self.gripper_open_threshold = 0.1
        # Cache per-joint limits for normalized gripper command mapping.
        self.gripper_joint_limits = {}
        for jn in self.gripper_joint_names:
            try:
                jr = self.env.model.joint(jn).range
                self.gripper_joint_limits[jn] = (float(jr[0]), float(jr[1]))
            except Exception:
                self.gripper_joint_limits[jn] = (0.0, 1.0)
        self.nova5_gripper_mode = (
            ('finger_joint' in joint_names_all)
            and ('right_outer_knuckle_joint' in joint_names_all)
        )
        self.nova5_gripper_mimic = {
            'finger_joint': 1.0,
            'left_inner_knuckle_joint': 1.0,
            'left_inner_finger_joint': -1.0,
            'right_inner_knuckle_joint': -1.0,
            'right_inner_finger_joint': 1.0,
            'right_outer_knuckle_joint': -1.0,
        }

        # Camera names with fallback.
        self.agent_cam_name = 'agentview' if 'agentview' in cam_names else None
        if self.agent_cam_name is None and len(cam_names) > 0:
            self.agent_cam_name = sorted(list(cam_names))[0]
        self.wrist_cam_name = 'egocentric' if 'egocentric' in cam_names else self.agent_cam_name
        self.side_cam_name = 'sideview' if 'sideview' in cam_names else self.agent_cam_name

        # Optional actuator-control mode (avoid kinematic qpos overwrite for tendon grippers).
        self.use_actuator_ctrl_mode = False
        self.ctrl_cmd = None
        self.ctrl_name_to_idx = {}
        self.arm_actuator_names = []
        self.gripper_actuator_name = None
        self.gripper_actuator_ctrlrange = (0.0, 1.0)
        ctrl_names = self._safe_names(getattr(self.env, 'ctrl_names', []))
        for i, cn in enumerate(ctrl_names):
            self.ctrl_name_to_idx[cn] = i
        if (self.ee_body_name == 'xarm_gripper_base_link') and ('gripper' in self.ctrl_name_to_idx):
            self.arm_actuator_names = [f'act{i+1}' for i in range(len(self.joint_names))]
            if all((an in self.ctrl_name_to_idx) for an in self.arm_actuator_names):
                self.use_actuator_ctrl_mode = True
                self.gripper_actuator_name = 'gripper'
                try:
                    cr = self.env.model.actuator(self.gripper_actuator_name).ctrlrange
                    self.gripper_actuator_ctrlrange = (float(cr[0]), float(cr[1]))
                except Exception:
                    self.gripper_actuator_ctrlrange = (0.0, 255.0)

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        self.env.init_viewer(
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    def reset(self, seed = None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed is not None:
            np.random.seed(seed=seed)
        q_init = np.zeros(len(self.joint_names), dtype=np.float32)
        # Robot-specific home pose for safer initialization.
        if self.ee_body_name in ('Link6', 'link6', 'xarm_gripper_base_link'):
            # Nova5: keep gripper pointing down and away from objects.
            init_p = np.array([0.24, 0.0, 1.15], dtype=np.float32)
            init_rpy_deg = [180.0, 0.0, 0.0]
        else:
            # Legacy OMY home.
            init_p = np.array([0.3, 0.0, 1.0], dtype=np.float32)
            init_rpy_deg = [90.0, -0.0, 90.0]

        q_zero,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = self.ee_body_name,
            q_init       = q_init, # ik from zero pose
            p_trgt       = init_p,
            R_trgt       = rpy2r(np.deg2rad(init_rpy_deg)),
        )
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)

        # Set object positions
        obj_names = self.env.get_body_names(prefix='body_obj_')
        n_obj = len(obj_names)
        if self.ee_body_name == 'Link6':
            # Nova5: spawn objects further forward to avoid startup collision.
            x_range = [+0.48, +0.65]
            y_range = [-0.18, +0.18]
            min_dist = 0.16
        else:
            x_range = [+0.24, +0.4]
            y_range = [-0.2, +0.2]
            min_dist = 0.2
        obj_xyzs = sample_xyzs(
            n_obj,
            x_range   = x_range,
            y_range   = y_range,
            z_range   = [0.82,0.82],
            min_dist  = min_dist,
            xy_margin = 0.0
        )
        for obj_idx in range(n_obj):
            self.env.set_p_base_body(body_name=obj_names[obj_idx],p=obj_xyzs[obj_idx,:])
            self.env.set_R_base_body(body_name=obj_names[obj_idx],R=np.eye(3,3))
        self.env.forward(increase_tick=False)

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)
        self.ik_rest_pose = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, np.array([0.0] * len(self.gripper_joint_names), dtype=np.float32)])
        if self.use_actuator_ctrl_mode:
            ctrl = np.zeros(getattr(self.env, 'n_ctrl', 0), dtype=np.float32)
            for i, an in enumerate(self.arm_actuator_names):
                ai = self.ctrl_name_to_idx.get(an, None)
                if ai is not None and i < len(q_zero):
                    ctrl[ai] = float(q_zero[i])
            gi = self.ctrl_name_to_idx.get(self.gripper_actuator_name, None)
            if gi is not None:
                lo, hi = self.gripper_actuator_ctrlrange
                ctrl[gi] = float(lo)
            self.ctrl_cmd = ctrl
        self.p0, self.R0 = self.env.get_pR_body(body_name=self.ee_body_name)
        mug_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate([mug_init_pose, plate_init_pose],dtype=np.float32)
        for _ in range(100):
            self.step_env()
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.gripper_cmd_scalar = 0.0
        self.past_chars = []

    def step(self, action):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]

        '''
        if self.action_type == 'eef_pose':
            action = np.asarray(action, dtype=np.float32)
            motion = action[:6]
            has_motion = np.any(np.abs(motion) > self.action_deadband)

            # On zero motion, keep last commanded joints to avoid IK solution hopping.
            if not has_motion:
                if hasattr(self, 'compute_q'):
                    q = copy.deepcopy(self.compute_q)
                else:
                    q = self.env.get_qpos_joints(joint_names=self.joint_names)
            else:
                q = self.env.get_qpos_joints(joint_names=self.joint_names)
                self.p0 += motion[:3]
                self.R0 = self.R0.dot(rpy2r(motion[3:6]))
                if self.ik_backend == 'pybullet':
                    try:
                        q = self._solve_ik_pybullet(q_seed=q, p_trgt=self.p0, R_trgt=self.R0)
                    except Exception:
                        q, ik_err_stack, ik_info = solve_ik(
                            env                = self.env,
                            joint_names_for_ik = self.joint_names,
                            body_name_trgt     = self.ee_body_name,
                            q_init             = q,
                            p_trgt             = self.p0,
                            R_trgt             = self.R0,
                            max_ik_tick        = self.ik_max_tick,
                            ik_stepsize        = self.ik_stepsize,
                            ik_eps             = self.ik_eps,
                            ik_th              = self.ik_trim_th,
                            render             = False,
                            verbose_warning    = False,
                        )
                else:
                    q, ik_err_stack, ik_info = solve_ik(
                        env                = self.env,
                        joint_names_for_ik = self.joint_names,
                        body_name_trgt     = self.ee_body_name,
                        q_init             = q,
                        p_trgt             = self.p0,
                        R_trgt             = self.R0,
                        max_ik_tick        = self.ik_max_tick,
                        ik_stepsize        = self.ik_stepsize,
                        ik_eps             = self.ik_eps,
                        ik_th              = self.ik_trim_th,
                        render             = False,
                        verbose_warning    = False,
                    )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')
        
        # Map normalized gripper input [0,1] to each joint's physical range.
        g_target = float(np.clip(action[-1], 0.0, 1.0))
        if not hasattr(self, 'gripper_cmd_scalar'):
            self.gripper_cmd_scalar = g_target
        # xarm7 contact-aware close guard: if fingers are already in contact, do not keep closing.
        if self.xarm7_contact_stop_close and (self.ee_body_name == 'xarm_gripper_base_link'):
            is_closing = (g_target > self.gripper_cmd_scalar + 1e-6)
            if is_closing and self._xarm7_has_gripper_contact():
                g_target = self.gripper_cmd_scalar
        dg = g_target - self.gripper_cmd_scalar
        dg = np.clip(dg, -self.gripper_rate_per_step, self.gripper_rate_per_step)
        self.gripper_cmd_scalar = float(np.clip(self.gripper_cmd_scalar + dg, 0.0, 1.0))
        g = self.gripper_cmd_scalar
        gripper_cmd = []
        if self.nova5_gripper_mode and ('finger_joint' in self.gripper_joint_limits):
            f_lo, f_hi = self.gripper_joint_limits['finger_joint']
            finger_cmd = f_lo + g * (f_hi - f_lo)
            for jn in self.gripper_joint_names:
                lo, hi = self.gripper_joint_limits.get(jn, (0.0, 1.0))
                mult = self.nova5_gripper_mimic.get(jn, 1.0)
                v = float(mult * finger_cmd)
                gripper_cmd.append(float(np.clip(v, lo, hi)))
        else:
            for jn in self.gripper_joint_names:
                lo, hi = self.gripper_joint_limits.get(jn, (0.0, 1.0))
                gripper_cmd.append(lo + g * (hi - lo))
        gripper_cmd = np.array(gripper_cmd, dtype=np.float32)
        # Legacy OMY scale pattern for 4-finger setup.
        if len(self.gripper_joint_names) == 4:
            gripper_cmd[[1,3]] *= 0.8
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd]) if len(gripper_cmd) > 0 else q

        self.q = q
        # Build actuator control command when available (xarm7 tendon gripper).
        if self.use_actuator_ctrl_mode:
            ctrl = np.zeros(getattr(self.env, 'n_ctrl', 0), dtype=np.float32)
            for i, an in enumerate(self.arm_actuator_names):
                ai = self.ctrl_name_to_idx.get(an, None)
                if ai is not None and i < len(self.compute_q):
                    ctrl[ai] = float(self.compute_q[i])
            gi = self.ctrl_name_to_idx.get(self.gripper_actuator_name, None)
            if gi is not None:
                lo, hi = self.gripper_actuator_ctrlrange
                ctrl[gi] = float(lo + g * (hi - lo))
            self.ctrl_cmd = ctrl

        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def get_commanded_joint_action(self):
        '''
        Return the target joint action most recently prepared by step().

        This is distinct from get_joint_state(), which reads the current
        physical joint state from MuJoCo.
        '''
        if not hasattr(self, 'compute_q'):
            raise RuntimeError('Call step(action) before reading the commanded action')
        return np.concatenate([
            np.asarray(self.compute_q, dtype=np.float32),
            np.asarray([self.gripper_cmd_scalar], dtype=np.float32),
        ])

    def step_env(self):
        # Prefer actuator-control path when available (xarm7 tendon gripper).
        if self.use_actuator_ctrl_mode and (self.ctrl_cmd is not None) and (len(self.ctrl_cmd) == getattr(self.env, 'n_ctrl', 0)):
            self.env.step(self.ctrl_cmd)
            return

        # If no actuator exists OR control dimension mismatches this env wrapper,
        # set qpos directly and still advance one physics step.
        if (getattr(self.env, 'n_ctrl', 0) <= 0) or (len(self.q) != getattr(self.env, 'n_ctrl', 0)):
            qpos_idxs = self.env.get_idxs_fwd(self.control_joint_names)
            qvel_idxs = self.env.get_idxs_jac(self.control_joint_names)
            # Pre-hold arm/gripper before stepping.
            self.env.data.qpos[qpos_idxs] = self.q
            self.env.data.qvel[qvel_idxs] = 0.0
            self.env.forward(
                q=self.q,
                joint_names=self.control_joint_names,
                increase_tick=False,
            )
            self.env.step(ctrl=None, increase_tick=True)
            # Post-hold again to remove one-step drift/jitter from passive dynamics.
            self.env.data.qpos[qpos_idxs] = self.q
            self.env.data.qvel[qvel_idxs] = 0.0
            self.env.forward(increase_tick=False)
        else:
            self.env.step(self.q)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name=self.agent_cam_name)
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name=self.wrist_cam_name)
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name=self.side_cam_name)
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name=self.ee_body_name)
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        
        try:
            if (rgb_agent_view is not None) and (rgb_agent_view.size > 0):
                self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
            if (rgb_egocentric_view is not None) and (rgb_egocentric_view.size > 0):
                self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        except Exception:
            # Some notebook/headless backends can report zero overlay canvas briefly.
            pass
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
        self.env.render()

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        if self.gripper_monitor_joint is not None:
            gripper = self.env.get_qpos_joint(self.gripper_monitor_joint)
            gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        else:
            gripper_cmd = 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)
    
    def teleop_robot(self):
        '''
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation
        
        Keys:
            ---------     -----------------------
               w       ->        backward
            s  a  d        left   forward   right
            ---------      -----------------------
            In x, y plane

            ---------
            R: Moving Up
            F: Moving Down
            ---------
            In z axis

            ---------
            Q: Tilt left
            E: Tilt right
            UP: Look Upward
            Down: Look Donward
            Right: Turn right
            Left: Turn left
            ---------
            For rotation

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------   


        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        dp = self.teleop_pos_step
        dr = self.teleop_rot_step
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([dp,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-dp,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0,-dp,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0,dp,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0,0.0,dp])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0,0.0,-dp])
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=dr, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-dr, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=dr, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-dr, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=dr, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-dr, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action, False
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        if self.gripper_monitor_joint is not None:
            gripper = self.env.get_qpos_joint(self.gripper_monitor_joint)
            gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        else:
            gripper_cmd = 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def check_success(self):
        '''
        ['body_obj_mug_5', 'body_obj_plate_11']
        Check if the mug is placed on the plate
        + Gripper should be open and move upward above 0.9
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        xy_dist = float(np.linalg.norm(p_mug[:2] - p_plate[:2]))
        z_dist = float(abs(p_mug[2] - p_plate[2]))
        ee_z = float(self.env.get_p_body(self.ee_body_name)[2])

        gripper_open_norm = self._get_gripper_open_norm()
        gripper_open = (gripper_open_norm >= self.success_gripper_open_norm_threshold)

        placed = (xy_dist < self.success_xy_threshold) and (z_dist < self.success_z_threshold)
        lifted = (ee_z > self.success_ee_height_threshold)
        released = (not self.success_require_gripper_open) or gripper_open
        return bool(placed and lifted and released)

    def _xarm7_has_gripper_contact(self):
        """Detect contact between xarm7 finger bodies and mug body."""
        try:
            contact_pairs = self.env.get_contact_body_names()
        except Exception:
            return False
        finger_bodies = {
            'left_finger', 'right_finger',
            'left_outer_knuckle', 'right_outer_knuckle',
        }
        mug_bodies = {'object_mug_5', 'body_obj_mug_5'}
        for b1, b2 in contact_pairs:
            s = {b1, b2}
            if (len(s & finger_bodies) > 0) and (len(s & mug_bodies) > 0):
                return True
        return False

    def _get_gripper_open_norm(self):
        """
        Return normalized gripper openness in [0,1], where:
            1.0 = fully open, 0.0 = fully closed
        """
        if self.gripper_monitor_joint is None:
            return 1.0
        try:
            q = float(self.env.get_qpos_joint(self.gripper_monitor_joint)[0])
            lo, hi = self.gripper_joint_limits.get(self.gripper_monitor_joint, (0.0, 1.0))
            lo2, hi2 = (lo, hi) if lo <= hi else (hi, lo)
            denom = max(hi2 - lo2, 1e-9)
            close_norm = float(np.clip((q - lo2) / denom, 0.0, 1.0))
            open_norm = 1.0 - close_norm
            return float(np.clip(open_norm, 0.0, 1.0))
        except Exception:
            return 1.0

    def get_success_debug(self):
        """
        Diagnostic values for success condition.
        """
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        xy_dist = float(np.linalg.norm(p_mug[:2] - p_plate[:2]))
        z_dist = float(abs(p_mug[2] - p_plate[2]))
        ee_z = float(self.env.get_p_body(self.ee_body_name)[2])
        gripper_open_norm = self._get_gripper_open_norm()
        return {
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "ee_z": ee_z,
            "gripper_open_norm": gripper_open_norm,
            "success_xy_threshold": self.success_xy_threshold,
            "success_z_threshold": self.success_z_threshold,
            "success_ee_height_threshold": self.success_ee_height_threshold,
            "success_gripper_open_norm_threshold": self.success_gripper_open_norm_threshold,
            "success_require_gripper_open": self.success_require_gripper_open,
            "success": self.check_success(),
        }
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        return p_mug, p_plate
    
    def set_obj_pose(self, p_mug, p_plate):
        '''
        Set the object poses
        args:
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=p_mug)
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()


    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name=self.ee_body_name)
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)
