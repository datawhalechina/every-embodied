"""Send velocity commands to the SONIC G1 MuJoCo configuration."""

from pathlib import Path

import numpy as np
from hirobot import Action, ControlMode, make_robot


def main():
    config = Path(__file__).with_name("g1_sonic.yaml")
    robot = make_robot(str(config))
    if not robot.simulated:
        raise RuntimeError("This example only runs on a simulated robot")
    robot.connect()
    try:
        walk = Action(
            mode=ControlMode.BASE_VELOCITY,
            base_velocity=np.array([0.4, 0.0, 0.0]),
        )
        for _ in range(250):
            robot.send_action(walk)
        print("base position:", robot.get_observation().base_position)
        print("controller status:", robot.get_status().extra.get("wbc"))
        robot.hold()
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
