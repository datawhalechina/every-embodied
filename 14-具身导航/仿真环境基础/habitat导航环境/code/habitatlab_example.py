import os
from typing import TYPE_CHECKING, Union, cast

import git
import matplotlib.pyplot as plt
import numpy as np

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from habitat_sim.utils import viz_utils as vut

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

output_path = "examples/tutorials/habitat_lab_visualization/"
os.makedirs(output_path, exist_ok=True)

class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass


def example_top_down_map_measure():
    # Create habitat config
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Create ShortestPathFollowerAgent agent
        agent = ShortestPathFollowerAgent(
            env=env,
            goal_radius=config.habitat.task.measurements.success.success_distance,
        )
        # Create video of agent navigating in the first episode
        num_episodes = 1
        for _ in range(num_episodes):
            # Load the first episode and reset agent
            observations = env.reset()
            agent.reset()

            # Get metrics
            info = env.get_metrics()
            # Concatenate RGB-D observation and topdowm map into one image
            frame = observations_to_image(observations, info)

            # Remove top_down_map from metrics
            info.pop("top_down_map")
            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, info)
            # Add fame to vis_frames
            vis_frames = [frame]

            # Repeat the steps above while agent doesn't reach the goal
            while not env.episode_over:
                # Get the next best action
                action = agent.act(observations)
                if action is None:
                    break

                # Step in the environment
                observations = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(observations, info)

                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            current_episode = env.current_episode
            video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
            # Create video from images and save to disk
            images_to_video(
                vis_frames, output_path, video_name, fps=6, quality=9
            )
            vis_frames.clear()
            # Display video
            vut.display_video(f"{output_path}/{video_name}.mp4")

if __name__ == "__main__":
    example_top_down_map_measure()