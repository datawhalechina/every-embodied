"""Run a toy robot and an echo whole-body controller without hardware or weights."""

import numpy as np

from hirobot import Action, ControlMode, make_robot
from hiwbc.controllers.echo import EchoController
from hiwbc.robot import WbcRobot


def main():
    names = [f"j{i}" for i in range(4)]
    config = {
        "assembly": {
            "name": "tutorial-toy",
            "parts": {
                "body": {
                    "device": "body",
                    "control_hz": 50,
                    "joints": [{"name": name} for name in names],
                    "driver": {"type": "toy"},
                }
            },
        }
    }
    inner = make_robot(config)
    controller = EchoController(
        num_actions=4,
        joint_names=names,
        default_pos=[0.1, -0.1, 0.2, -0.2],
        stand_command=(0.0, 0.0, 0.0),
    )
    robot = WbcRobot(inner, controller=controller, start_seconds=0.1)
    robot.connect()
    try:
        print("simulated:", robot.simulated)
        print("initial joint positions:", robot.get_observation().joint_pos)
        command = Action(
            mode=ControlMode.BASE_VELOCITY,
            base_velocity=np.array([0.3, 0.0, 0.0]),
        )
        for _ in range(10):
            robot.send_action(command)
        print("joint positions after 10 ticks:", robot.get_observation().joint_pos)
        print("controller state:", robot.get_status().extra["wbc"])
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
