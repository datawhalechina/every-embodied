"""High-fidelity 1:1 MuJoCo MJCF scene builder for TuntunClaw Digital Twin.

Faithfully matches the real booth setup captured via overhead Orbbec & front C931:
- Desk surface at Z=0.000 (1.18m wide x 0.67m deep light oak exhibition tabletop)
- myCobot 280 Pi at (0, 0) on a 4-corner suction plate with 12:00 (-X) cable plugs
- Real 1:1 black EVA foam pads mounted on the adaptive gripper fingertips
- Tilted left red picking tray (with 2x 雀巢脆脆鲨 + 3x 舒客+小牙膏 compound bodies)
- Center walnut wooden delivery tray
- Top-right D-Robotics RDK X5 Magic Box with glowing cyan LED array & acrylic bracket
- Left keyboard, mouse, angled 13.3" portable display; right-side fruit plate & vials
- 1:1 calibrated overhead_top, c931_view, booth_overview, side_view, magicbox_cam, wrist_rgb
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np


def _yaw_quat(deg: float) -> str:
    r = math.radians(deg) * 0.5
    return f"{math.cos(r):.6f} 0.000000 0.000000 {math.sin(r):.6f}"


def _lookat_xyaxes(pos: tuple[float, float, float], target: tuple[float, float, float], up=(0.0, 0.0, 1.0)) -> str:
    fwd = np.array(target, dtype=np.float64) - np.array(pos, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    z_cam = -fwd
    x_cam = np.cross(fwd, np.array(up, dtype=np.float64))
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    return (
        f"{x_cam[0]:.4f} {x_cam[1]:.4f} {x_cam[2]:.4f} "
        f"{y_cam[0]:.4f} {y_cam[1]:.4f} {y_cam[2]:.4f}"
    )


def _wafer_body_xml(idx: int, x: float, y: float, long_az_deg: float) -> str:
    # 15g Nestle Cuicuisha wafer: ~74mm x 25mm x 17mm, center at Z=0.0125 on 4mm tray floor
    q = _yaw_quat(long_az_deg)
    return f"""    <body name="obj_cuicuisha_{idx}" pos="{x:.4f} {y:.4f} 0.0128" quat="{q}">
      <freejoint name="fj_cuicuisha_{idx}"/>
      <inertial pos="0 0 0" mass="0.016" diaginertia="0.000006 0.000018 0.000018"/>
      <geom name="g_cuicuisha_{idx}_core" type="box" size="0.034 0.0122 0.0082" material="mat_wafer_body" friction="1.45 0.02 0.01"/>
      <geom name="g_cuicuisha_{idx}_band" type="box" pos="0.003 0 0.0006" size="0.020 0.0124 0.0078" material="mat_wafer_label" contype="0" conaffinity="0"/>
      <geom name="g_cuicuisha_{idx}_stripe" type="box" pos="-0.004 0 0.0084" size="0.014 0.0055 0.0004" material="mat_wafer_gold" contype="0" conaffinity="0"/>
      <geom name="g_cuicuisha_{idx}_seal1" type="box" pos="0.037 0 -0.001" size="0.0035 0.0115 0.0022" material="mat_wafer_seal" contype="0" conaffinity="0"/>
      <geom name="g_cuicuisha_{idx}_seal2" type="box" pos="-0.037 0 -0.001" size="0.0035 0.0115 0.0022" material="mat_wafer_seal" contype="0" conaffinity="0"/>
    </body>"""


def _toothpaste_body_xml(idx: int, cap_x: float, cap_y: float, long_az_deg: float) -> str:
    # 20g Saky+ Travel Toothpaste: origin at white screw cap center so body extends toward -X_local.
    # White cap: Ø14.2mm x 16mm tall upright cylinder lying horizontally (axis along X? No: cap axis is along tube X -> box 0.008 x 0.0072 x 0.0075 works great & rolls less!).
    # Wait: horizontal cylinder / rounded octagonal cap along X_local with slight anti-roll flat bottom!
    q = _yaw_quat(long_az_deg)
    return f"""    <body name="obj_toothpaste_{idx}" pos="{cap_x:.4f} {cap_y:.4f} 0.0116" quat="{q}">
      <freejoint name="fj_toothpaste_{idx}"/>
      <inertial pos="-0.030 0 -0.0005" mass="0.020" diaginertia="0.000005 0.000022 0.000022"/>
      <!-- Graspable cylindrical-profile white cap at head (15.5mm wide x 15mm tall) -->
      <geom name="g_tp_{idx}_cap" type="box" pos="0.000 0 0.000" size="0.0085 0.0074 0.0074" material="mat_tp_cap" friction="1.55 0.02 0.01"/>
      <geom name="g_tp_{idx}_cap_tip" type="cylinder" pos="0.000 0 0" quat="0.707107 0 0.707107 0" size="0.0072 0.0088" material="mat_tp_cap" contype="0" conaffinity="0"/>
      <!-- Sky-teal brand band around tube neck -->
      <geom name="g_tp_{idx}_neck" type="box" pos="-0.024 0 -0.0005" size="0.017 0.0095 0.0066" material="mat_tp_teal" friction="1.15 0.02 0.01"/>
      <!-- White tapered flat body & crimped silver-white tail -->
      <geom name="g_tp_{idx}_body" type="box" pos="-0.052 0 -0.0018" size="0.016 0.0125 0.0048" material="mat_tp_white" friction="1.10 0.02 0.01"/>
      <geom name="g_tp_{idx}_crimp" type="box" pos="-0.071 0 -0.0040" size="0.004 0.0132 0.0020" material="mat_tp_tail" contype="0" conaffinity="0"/>
    </body>"""


def build_mjcf(out_path: Path | None = None) -> None:
    if out_path is None:
        out_path = Path(__file__).resolve().parent / "mycobot_280_scene.xml"

    c931_xyaxes = _lookat_xyaxes((0.62, 0.72, 0.48), (0.08, 0.04, 0.06))
    booth_xyaxes = _lookat_xyaxes((0.64, -0.32, 0.46), (0.09, 0.03, 0.06))
    side_xyaxes = _lookat_xyaxes((0.50, -0.04, 0.28), (0.08, 0.02, 0.08))
    mb_cam_xyaxes = _lookat_xyaxes((0.055, 0.0, 0.022), (-0.08, -0.25, 0.01))
    wrist_xyaxes = _lookat_xyaxes((0.048, 0.052, 0.0), (0.0, 0.195, 0.0), up=(0.0, 0.0, 1.0))

    # Calibrated real positions in rear-left Red Tray from after_teach_top.jpg (all r in [150..215]mm):
    wafers_xml = "\n".join([
        _wafer_body_xml(1, 0.022, -0.170, 88.0),
        _wafer_body_xml(2, 0.064, -0.168, 82.0),
    ])
    # Toothpastes: well-spaced horizontal tubes with white screw caps right (+Y_world) and tails left (-Y_world)
    pastes_xml = "\n".join([
        _toothpaste_body_xml(1, 0.096, -0.112, 84.0),
        _toothpaste_body_xml(2, 0.132, -0.112, 90.0),
        _toothpaste_body_xml(3, 0.168, -0.112, 95.0),
    ])

    red_tray_quat = _yaw_quat(0.0)
    monitor_quat = _yaw_quat(165.0)
    mb_quat = _yaw_quat(4.0)

    xml = f"""<mujoco model="mycobot_280_pi_tuntuncloud">
  <compiler angle="radian" meshdir="assets/mycobot_280/stl" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="6"/>

  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight diffuse="0.38 0.38 0.38" ambient="0.24 0.24 0.24" specular="0.16 0.16 0.16"/>
    <rgba haze="0.93 0.94 0.95 1"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <!-- Official Elephant Robotics myCobot 280 Pi + Camera Flange + Adaptive Gripper STL meshes -->
    <mesh name="G_base" file="G_base.stl"/>
    <mesh name="joint1_pi" file="joint1_pi.stl"/>
    <mesh name="joint2" file="joint2.stl"/>
    <mesh name="joint3" file="joint3.stl"/>
    <mesh name="joint4" file="joint4.stl"/>
    <mesh name="joint5" file="joint5.stl"/>
    <mesh name="joint6" file="joint6.stl"/>
    <mesh name="joint7" file="joint7.stl"/>
    <mesh name="camera_flange" file="camera_flange.stl"/>
    <mesh name="gripper_base" file="gripper_base.stl"/>
    <mesh name="gripper_left1" file="gripper_left1.stl"/>
    <mesh name="gripper_left2" file="gripper_left2.stl"/>
    <mesh name="gripper_left3" file="gripper_left3.stl"/>
    <mesh name="gripper_right1" file="gripper_right1.stl"/>
    <mesh name="gripper_right2" file="gripper_right2.stl"/>
    <mesh name="gripper_right3" file="gripper_right3.stl"/>

    <!-- High-fidelity room & booth surface materials -->
    <texture name="tex_wood" type="2d" builtin="gradient" rgb1="0.72 0.57 0.41" rgb2="0.65 0.50 0.35" width="256" height="256"/>
    <texture name="tex_floor" type="2d" builtin="checker" rgb1="0.82 0.82 0.80" rgb2="0.78 0.78 0.76" width="256" height="256"/>
    <texture name="tex_sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.97" rgb2="0.88 0.90 0.92" width="256" height="256"/>

    <material name="mat_wood" texture="tex_wood" texrepeat="6 4" specular="0.04" shininess="0.10" roughness="0.82"/>
    <material name="mat_floor" texture="tex_floor" texrepeat="12 12" specular="0.04" roughness="0.85"/>
    <material name="mat_wall" rgba="0.94 0.94 0.93 1" specular="0.03" roughness="0.9"/>
    <material name="mat_robot_white" rgba="0.95 0.96 0.97 1" specular="0.35" shininess="0.5"/>
    <material name="mat_robot_grey" rgba="0.55 0.58 0.61 1" specular="0.22" shininess="0.32"/>
    <material name="mat_robot_dark" rgba="0.14 0.16 0.18 1" specular="0.3" shininess="0.4"/>
    <material name="mat_foam_pad" rgba="0.09 0.09 0.10 1" specular="0.04" roughness="0.92"/>
    <material name="mat_suction_plate" rgba="0.62 0.64 0.67 0.88" specular="0.3"/>
    <material name="mat_suction_cup" rgba="0.82 0.86 0.90 0.55" specular="0.6" shininess="0.7"/>
    <material name="mat_atom_green" rgba="0.10 0.90 0.35 1" emission="0.65"/>

    <!-- Trays matching real red plastic & dark walnut wood tray -->
    <material name="mat_red_tray" rgba="0.78 0.09 0.10 1" specular="0.35" shininess="0.45"/>
    <material name="mat_red_tray_in" rgba="0.72 0.07 0.09 1" specular="0.28" shininess="0.35"/>
    <material name="mat_place_tray" rgba="0.16 0.10 0.07 1" specular="0.12"/>
    <material name="mat_place_inner" rgba="0.20 0.13 0.09 1" specular="0.10"/>

    <!-- D-Robotics RDK X5 Magic Box & Cyan Glow -->
    <material name="mat_magicbox_body" rgba="0.11 0.12 0.14 1" specular="0.38" shininess="0.5"/>
    <material name="mat_magicbox_acrylic" rgba="0.78 0.88 0.90 0.42" specular="0.8"/>
    <material name="mat_magicbox_led" rgba="0.08 0.95 0.86 1" emission="1.0"/>
    <material name="mat_monitor_frame" rgba="0.09 0.10 0.12 1" specular="0.32"/>
    <material name="mat_monitor_screen" rgba="0.78 0.80 0.79 1" emission="0.18"/>
    <material name="mat_kb" rgba="0.11 0.12 0.13 1" specular="0.18"/>

    <!-- Real booth SKU materials: 雀巢脆脆鲨 & 舒客+小牙膏 -->
    <material name="mat_wafer_body" rgba="0.13 0.04 0.04 1" specular="0.16" shininess="0.30"/>
    <material name="mat_wafer_label" rgba="0.22 0.04 0.04 1" specular="0.16"/>
    <material name="mat_wafer_gold" rgba="0.46 0.34 0.08 1" specular="0.18"/>
    <material name="mat_wafer_seal" rgba="0.14 0.04 0.04 1" specular="0.14"/>
    <material name="mat_tp_cap" rgba="0.98 0.99 1.00 1" specular="0.32" shininess="0.4"/>
    <material name="mat_tp_white" rgba="0.80 0.82 0.85 1" specular="0.20"/>
    <material name="mat_tp_teal" rgba="0.36 0.58 0.72 1" specular="0.22"/>
    <material name="mat_tp_tail" rgba="0.72 0.75 0.78 1" specular="0.25"/>

    <!-- Right-side calibration props (dish, fruits, vials, coasters) -->
    <material name="mat_apple" rgba="0.78 0.14 0.14 1" specular="0.35"/>
    <material name="mat_orange" rgba="0.96 0.54 0.05 1" specular="0.3"/>
    <material name="mat_pear" rgba="0.86 0.80 0.16 1" specular="0.3"/>
    <material name="mat_dish" rgba="0.94 0.94 0.95 1" specular="0.4"/>
    <material name="mat_vial" rgba="0.12 0.10 0.09 1" specular="0.4"/>
    <material name="mat_vial_cap" rgba="0.14 0.56 0.62 1" specular="0.3"/>
    <material name="mat_coaster" rgba="0.88 0.84 0.78 0.85" specular="0.15"/>
  </asset>

  <default>
    <joint damping="1.5" armature="0.02"/>
    <geom contype="1" conaffinity="1" condim="4" friction="1.1 0.015 0.008"/>
  </default>

  <worldbody>
    <!-- Realistic Indoor Studio Lighting + Magic Box Cyan Accent -->
    <light pos="0.22 0.14 1.38" dir="0 0 -1" diffuse="0.58 0.57 0.54" specular="0.14 0.14 0.14" castshadow="true"/>
    <light pos="0.45 -0.45 0.95" dir="-0.3 0.35 -1" diffuse="0.28 0.29 0.31" castshadow="false"/>
    <light pos="-0.09 0.30 0.12" dir="0 0 -1" diffuse="0.08 0.42 0.38" attenuation="1 2.5 5" castshadow="false"/>

    <!-- 1:1 Aligned Real Booth Cameras (matched to empirical clamped teach calibration) -->
    <camera name="overhead_top" pos="0.351 0.188 0.960" xyaxes="0 1 0 -1 0 0" fovy="48"/>
    <camera name="c931_view" pos="0.62 0.72 0.48" xyaxes="{c931_xyaxes}" fovy="46"/>
    <camera name="booth_overview" pos="0.64 -0.32 0.46" xyaxes="{booth_xyaxes}" fovy="46"/>
    <camera name="side_view" pos="0.50 -0.04 0.28" xyaxes="{side_xyaxes}" fovy="44"/>

    <!-- Room Floor & White Back Wall (directly behind clamped rear desk edge) -->
    <geom name="floor" type="plane" pos="0 0 -0.75" size="4 4 0.1" material="mat_floor" contype="0" conaffinity="0"/>
    <geom name="back_wall" type="box" pos="-0.095 0.160 0.25" size="0.02 2.2 1.1" material="mat_wall" contype="0" conaffinity="0"/>

    <!-- Oak Exhibition Desk: 1.18m wide (Y:-0.43..+0.75) x 0.67m deep (X:-0.075..+0.595), surface Z=0.000 -->
    <body name="desk" pos="0.260 0.160 -0.020">
      <geom name="desk_top" type="box" size="0.335 0.590 0.020" material="mat_wood"/>
      <geom name="leg_1" type="cylinder" pos="-0.29 -0.53 -0.37" size="0.022 0.35" material="mat_wood" contype="0" conaffinity="0"/>
      <geom name="leg_2" type="cylinder" pos="-0.29 0.53 -0.37" size="0.022 0.35" material="mat_wood" contype="0" conaffinity="0"/>
      <geom name="leg_3" type="cylinder" pos="0.29 -0.53 -0.37" size="0.022 0.35" material="mat_wood" contype="0" conaffinity="0"/>
      <geom name="leg_4" type="cylinder" pos="0.29 0.53 -0.37" size="0.022 0.35" material="mat_wood" contype="0" conaffinity="0"/>
    </body>

    <!-- Rear-Left Red Picking Tray (center=(+0.098,-0.168), yaw=0 deg, 252x197mm -> long along X_world) -->
    <body name="picking_zone" pos="0.098 -0.168 0.002" quat="{red_tray_quat}">
      <geom name="red_tray_bottom" type="box" size="0.120 0.092 0.002" material="mat_red_tray_in"/>
      <geom name="red_tray_w1" type="box" pos="0 0.094 0.007" size="0.124 0.004 0.0075" material="mat_red_tray"/>
      <geom name="red_tray_w2" type="box" pos="0 -0.094 0.007" size="0.124 0.004 0.0075" material="mat_red_tray"/>
      <geom name="red_tray_w3" type="box" pos="0.122 0 0.007" size="0.004 0.094 0.0075" material="mat_red_tray"/>
      <geom name="red_tray_w4" type="box" pos="-0.122 0 0.007" size="0.004 0.094 0.0075" material="mat_red_tray"/>
    </body>

    <!-- Rear-Right Walnut Wooden Delivery Tray (center=(+0.091,+0.190), size=301x196mm) -->
    <body name="delivery_tray" pos="0.091 0.190 0.002">
      <geom name="tray_bottom" type="box" size="0.146 0.094 0.002" material="mat_place_inner"/>
      <geom name="tray_wall_n" type="box" pos="-0.148 0 0.010" size="0.005 0.098 0.010" material="mat_place_tray"/>
      <geom name="tray_wall_s" type="box" pos="0.148 0 0.010" size="0.005 0.098 0.010" material="mat_place_tray"/>
      <geom name="tray_wall_w" type="box" pos="0 -0.096 0.010" size="0.150 0.005 0.010" material="mat_place_tray"/>
      <geom name="tray_wall_e" type="box" pos="0 0.096 0.010" size="0.150 0.005 0.010" material="mat_place_tray"/>
    </body>

    <!-- D-Robotics RDK X5 Magic Box (center=(+0.030,+0.405)) with glowing cyan LED -->
    <body name="magic_box" pos="0.030 0.405 0.026" quat="{mb_quat}">
      <geom name="mb_chassis" type="box" size="0.056 0.048 0.025" material="mat_magicbox_body"/>
      <geom name="mb_led_bar" type="box" pos="0.056 0 0.008" size="0.004 0.036 0.008" material="mat_magicbox_led" contype="0" conaffinity="0"/>
      <geom name="mb_bracket" type="box" pos="0.040 0 0.042" quat="0.92388 0 -0.38268 0" size="0.003 0.045 0.024" material="mat_magicbox_acrylic" contype="0" conaffinity="0"/>
      <camera name="magicbox_cam" pos="0.055 0 0.022" xyaxes="{mb_cam_xyaxes}" fovy="62"/>
    </body>

    <!-- Left Desktop Obstacles (Logitech Keyboard, Mouse, Angled 13.3" Portable Monitor) -->
    <body name="keyboard" pos="0.140 -0.340 0.009">
      <geom name="kb_body" type="box" size="0.215 0.068 0.009" material="mat_kb" contype="0" conaffinity="0"/>
    </body>
    <body name="mouse" pos="0.390 -0.365 0.014">
      <geom name="mouse_body" type="ellipsoid" size="0.048 0.030 0.014" material="mat_kb" contype="0" conaffinity="0"/>
    </body>
    <body name="portable_monitor" pos="0.415 -0.315 0.092" quat="{monitor_quat}">
      <geom name="mon_panel" type="box" size="0.005 0.145 0.088" material="mat_monitor_frame" contype="0" conaffinity="0"/>
      <geom name="mon_screen" type="box" pos="0.005 0 0" size="0.001 0.136 0.080" material="mat_monitor_screen" contype="0" conaffinity="0"/>
      <geom name="mon_stand" type="box" pos="-0.042 0 -0.072" size="0.045 0.065 0.003" material="mat_robot_dark" contype="0" conaffinity="0"/>
    </body>

    <!-- Right-side Exhibition Props (Coasters, Essential Vials, Fruit Ceramic Dish) -->
    <body name="right_props" pos="0.320 0.620 0.000">
      <geom name="coaster_1" type="cylinder" pos="0.14 -0.02 0.004" size="0.046 0.004" material="mat_coaster" contype="0" conaffinity="0"/>
      <geom name="coaster_2" type="cylinder" pos="0.14 0.10 0.004" size="0.046 0.004" material="mat_coaster" contype="0" conaffinity="0"/>
      <geom name="vial_1" type="cylinder" pos="-0.03 -0.02 0.025" size="0.014 0.025" material="mat_vial" contype="0" conaffinity="0"/>
      <geom name="vial_2" type="cylinder" pos="-0.03 0.02 0.025" size="0.014 0.025" material="mat_vial" contype="0" conaffinity="0"/>
      <geom name="vial_3" type="cylinder" pos="0.01 -0.02 0.025" size="0.014 0.025" material="mat_vial_cap" contype="0" conaffinity="0"/>
      <geom name="vial_4" type="cylinder" pos="0.01 0.02 0.025" size="0.014 0.025" material="mat_vial" contype="0" conaffinity="0"/>
      <geom name="vial_5" type="cylinder" pos="0.05 -0.02 0.025" size="0.014 0.025" material="mat_vial" contype="0" conaffinity="0"/>
      <geom name="vial_6" type="cylinder" pos="0.05 0.02 0.025" size="0.014 0.025" material="mat_vial_cap" contype="0" conaffinity="0"/>
      <geom name="fruit_dish" type="cylinder" pos="-0.04 0.10 0.006" size="0.072 0.006" material="mat_dish" contype="0" conaffinity="0"/>
      <geom name="prop_apple" type="sphere" pos="-0.07 0.11 0.042" size="0.032" material="mat_apple" contype="0" conaffinity="0"/>
      <geom name="prop_orange" type="sphere" pos="-0.02 0.08 0.040" size="0.030" material="mat_orange" contype="0" conaffinity="0"/>
      <geom name="prop_pear" type="sphere" pos="0.01 0.13 0.036" size="0.026" material="mat_pear" contype="0" conaffinity="0"/>
    </body>

    <!-- ================================================================ -->
    <!-- OFFICIAL 1:1 myCobot 280 Pi + 4-Suction Plate + 12:00 Plugs      -->
    <!-- Mounted at world origin (0, 0, 0), +X front (6:00), -Y Red Tray   -->
    <!-- ================================================================ -->
    <body name="g_base" pos="0 0 0.030">
      <!-- 4-corner suction base & 3:00 (+Y) cable plugs matching clamped desk hardware -->
      <geom name="suction_plate" type="box" pos="0 0 -0.022" size="0.060 0.060 0.004" material="mat_suction_plate" contype="0" conaffinity="0"/>
      <geom name="sc_1" type="cylinder" pos="0.048 0.048 -0.026" size="0.014 0.004" material="mat_suction_cup" contype="0" conaffinity="0"/>
      <geom name="sc_2" type="cylinder" pos="0.048 -0.048 -0.026" size="0.014 0.004" material="mat_suction_cup" contype="0" conaffinity="0"/>
      <geom name="sc_3" type="cylinder" pos="-0.048 0.048 -0.026" size="0.014 0.004" material="mat_suction_cup" contype="0" conaffinity="0"/>
      <geom name="sc_4" type="cylinder" pos="-0.048 -0.048 -0.026" size="0.014 0.004" material="mat_suction_cup" contype="0" conaffinity="0"/>
      <geom name="rear_plugs" type="box" pos="0 0.055 -0.005" size="0.034 0.008 0.014" material="mat_robot_dark" contype="0" conaffinity="0"/>
      <geom name="power_switch" type="box" pos="-0.022 0.058 -0.004" size="0.007 0.004 0.007" material="mat_red_tray" contype="0" conaffinity="0"/>

      <geom name="g_base_vis" type="mesh" mesh="G_base" pos="0 0 -0.030" quat="0.707105 0.000000 0.000000 0.707108" material="mat_robot_grey" contype="0" conaffinity="0"/>
      <body name="joint1" pos="0 0 0">
        <geom name="joint1_vis" type="mesh" mesh="joint1_pi" pos="0 0 0" material="mat_robot_grey" contype="0" conaffinity="0"/>
        <geom name="joint1_col" type="cylinder" pos="0 0 0.04" size="0.048 0.042" rgba="0 0 0 0"/>

        <body name="joint2" pos="0 0 0.13956">
          <inertial pos="0 0 -0.02" mass="0.22" diaginertia="0.0002 0.0002 0.0002"/>
          <joint name="joint1_j" type="hinge" axis="0 0 1" range="-2.9321 2.9321"/>
          <geom name="joint2_vis" type="mesh" mesh="joint2" pos="0 0 -0.06096" quat="0.707105 0.000000 0.000000 -0.707108" material="mat_robot_white" contype="0" conaffinity="0"/>

          <body name="joint3" pos="0 0 -0.001" quat="0.499998 0.500002 0.500000 -0.500000">
            <inertial pos="-0.055 0 0" mass="0.20" diaginertia="0.0003 0.0003 0.00015"/>
            <joint name="joint2_j" type="hinge" axis="0 0 1" range="-2.4434 2.4434"/>
            <geom name="joint3_vis" type="mesh" mesh="joint3" pos="0 0 0.03256" quat="0.707105 0.000000 -0.707108 0.000000" material="mat_robot_white" contype="0" conaffinity="0"/>

            <body name="joint4" pos="-0.1104 0 0">
              <inertial pos="-0.048 0 0.03" mass="0.18" diaginertia="0.00025 0.00025 0.00012"/>
              <joint name="joint3_j" type="hinge" axis="0 0 1" range="-2.6179 2.6179"/>
              <geom name="joint4_vis" type="mesh" mesh="joint4" pos="0 0 0.03056" quat="0.707105 0.000000 -0.707108 0.000000" material="mat_robot_white" contype="0" conaffinity="0"/>

              <body name="joint5" pos="-0.096 0 0.06462" quat="0.707105 0.000000 0.000000 -0.707108">
                <inertial pos="0 -0.035 0" mass="0.14" diaginertia="0.00015 0.00015 0.0001"/>
                <joint name="joint4_j" type="hinge" axis="0 0 1" range="-2.6179 2.6179"/>
                <geom name="joint5_vis" type="mesh" mesh="joint5" pos="0 -0.001 -0.0345" quat="0.499998 0.500002 -0.500000 0.500000" material="mat_robot_white" contype="0" conaffinity="0"/>

                <body name="joint6" pos="0 -0.07318 -0.001" quat="0.499998 0.500000 -0.500000 0.500002">
                  <inertial pos="0 0.02 0" mass="0.12" diaginertia="0.0001 0.0001 0.0001"/>
                  <joint name="joint5_j" type="hinge" axis="0 0 1" range="-2.7052 2.7925"/>
                  <geom name="joint6_vis" type="mesh" mesh="joint6" pos="0 0 -0.038" material="mat_robot_white" contype="0" conaffinity="0"/>
                  <geom name="atom_led" type="box" pos="0 -0.019 -0.015" size="0.009 0.0015 0.009" material="mat_atom_green" contype="0" conaffinity="0"/>

                  <body name="joint6_flange" pos="0 0.0456 0" quat="0.707105 -0.707108 0.000000 0.000000">
                    <inertial pos="0 0 0.02" mass="0.08" diaginertia="0.00005 0.00005 0.00005"/>
                    <joint name="joint6_j" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>
                    <geom name="joint7_vis" type="mesh" mesh="joint7" pos="0 0 -0.012" material="mat_robot_grey" contype="0" conaffinity="0"/>

                    <!-- Wrist RGB Camera Flange -->
                    <body name="camera_flange" pos="0 0 0.010" quat="0.707105 0.707108 0.000000 0.000000">
                      <geom name="camera_flange_vis" type="mesh" mesh="camera_flange" pos="0.024 -0.008 0" quat="0.000001 0.000000 1.000000 0.000000" material="mat_robot_white" contype="0" conaffinity="0"/>
                      <geom name="camera_lens" type="cylinder" pos="0.044 0.010 0" quat="0.707105 0.707108 0.000000 0.000000" size="0.006 0.004" material="mat_robot_dark" contype="0" conaffinity="0"/>
                      <camera name="wrist_rgb" pos="0.048 0.052 0.0" xyaxes="{wrist_xyaxes}" fovy="62"/>
                    </body>

                    <!-- Elephant Robotics Adaptive Gripper with Calibrated Black EVA Foam Pads -->
                    <body name="gripper_base" pos="0 0 0.04" quat="0.707105 0.707108 0 0">
                      <geom name="gripper_base_vis" type="mesh" mesh="gripper_base" pos="0 0 -0.012" quat="1 0 0 0" material="mat_robot_white" contype="0" conaffinity="0"/>
                      <site name="tcp_site" pos="0 0.072 0" size="0.003" rgba="1 0.2 0.2 0.0"/>

                      <body name="gripper_left3" pos="-0.012 0.005 0">
                        <inertial pos="-0.01 0.01 0" mass="0.018" diaginertia="0.00001 0.00001 0.00001"/>
                        <joint name="gripper_j" type="hinge" axis="0 0 1" range="-0.74 0.15"/>
                        <geom name="gl3_vis" type="mesh" mesh="gripper_left3" pos="0.012 0.0025 -0.012" quat="1 0 0 0" material="mat_robot_white" contype="0" conaffinity="0"/>
                        <body name="gripper_left1" pos="-0.027 0.016 0">
                          <inertial pos="0 0.025 0" mass="0.018" diaginertia="0.00001 0.00001 0.00001"/>
                          <joint name="gripper_l1_j" type="hinge" axis="0 0 1" range="-0.5 0.8"/>
                          <geom name="gl1_vis" type="mesh" mesh="gripper_left1" pos="0.039 -0.0133 -0.012" quat="1 0 0 0" material="mat_robot_white" contype="0" conaffinity="0"/>
                          <geom name="gl1_pad" type="box" pos="0.0175 0.040 0" size="0.0035 0.013 0.010" material="mat_foam_pad" friction="1.6 0.02 0.01"/>
                        </body>
                      </body>

                      <body name="gripper_left2" pos="-0.005 0.027 0">
                        <inertial pos="-0.005 0.01 0" mass="0.010" diaginertia="0.00001 0.00001 0.00001"/>
                        <joint name="gripper_l2_j" type="hinge" axis="0 0 1" range="-0.8 0.5"/>
                        <geom name="gl2_vis" type="mesh" mesh="gripper_left2" pos="0.005 -0.0195 -0.012" quat="1 0 0 0" material="mat_robot_white" contype="0" conaffinity="0"/>
                      </body>

                      <body name="gripper_right3" pos="0.012 0.005 0">
                        <inertial pos="0.01 0.01 0" mass="0.018" diaginertia="0.00001 0.00001 0.00001"/>
                        <joint name="gripper_r3_j" type="hinge" axis="0 0 1" range="-0.15 0.74"/>
                        <geom name="gr3_vis" type="mesh" mesh="gripper_right3" pos="-0.012 0.0025 -0.012" quat="1 0 0 0" material="mat_robot_white" contype="0" conaffinity="0"/>
                        <body name="gripper_right1" pos="0.027 0.016 0">
                          <inertial pos="0 0.025 0" mass="0.018" diaginertia="0.00001 0.00001 0.00001"/>
                          <joint name="gripper_r1_j" type="hinge" axis="0 0 1" range="-0.8 0.5"/>
                          <geom name="gr1_vis" type="mesh" mesh="gripper_right1" pos="-0.039 -0.0133 -0.012" quat="1 0 0 0" material="mat_robot_white" contype="0" conaffinity="0"/>
                          <geom name="gr1_pad" type="box" pos="-0.0175 0.040 0" size="0.0035 0.013 0.010" material="mat_foam_pad" friction="1.6 0.02 0.01"/>
                        </body>
                      </body>

                      <body name="gripper_right2" pos="0.005 0.027 0">
                        <inertial pos="0.005 0.01 0" mass="0.010" diaginertia="0.00001 0.00001 0.00001"/>
                        <joint name="gripper_r2_j" type="hinge" axis="0 0 1" range="-0.5 0.8"/>
                        <geom name="gr2_vis" type="mesh" mesh="gripper_right2" pos="-0.005 -0.0195 -0.012" quat="1 0 0 0" material="mat_robot_white" contype="0" conaffinity="0"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- ================================================================ -->
    <!-- GRASPABLE BOOTH SKUS (2x 雀巢脆脆鲨 + 3x 舒客+小牙膏)             -->
    <!-- ================================================================ -->
{wafers_xml}
{pastes_xml}
  </worldbody>

  <equality>
    <joint joint1="gripper_l2_j" joint2="gripper_j" polycoef="0 1 0 0 0"/>
    <joint joint1="gripper_l1_j" joint2="gripper_j" polycoef="0 -1 0 0 0"/>
    <joint joint1="gripper_r3_j" joint2="gripper_j" polycoef="0 -1 0 0 0"/>
    <joint joint1="gripper_r2_j" joint2="gripper_j" polycoef="0 -1 0 0 0"/>
    <joint joint1="gripper_r1_j" joint2="gripper_j" polycoef="0 1 0 0 0"/>
  </equality>

  <actuator>
    <position name="act_j1" joint="joint1_j" kp="85" kv="8" ctrlrange="-2.9321 2.9321"/>
    <position name="act_j2" joint="joint2_j" kp="85" kv="8" ctrlrange="-2.4434 2.4434"/>
    <position name="act_j3" joint="joint3_j" kp="85" kv="8" ctrlrange="-2.6179 2.6179"/>
    <position name="act_j4" joint="joint4_j" kp="95" kv="7" ctrlrange="-2.6179 2.6179"/>
    <position name="act_j5" joint="joint5_j" kp="55" kv="5" ctrlrange="-2.7052 2.7925"/>
    <position name="act_j6" joint="joint6_j" kp="45" kv="4" ctrlrange="-3.14159 3.14159"/>
    <position name="act_gripper" joint="gripper_j" kp="28" kv="2.5" ctrlrange="-0.74 0.15"/>
  </actuator>
</mujoco>
"""
    out_path.write_text(xml, encoding="utf-8")


if __name__ == "__main__":
    build_mjcf()
