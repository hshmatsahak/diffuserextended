import sys
import os
import numpy as np
import diffuser.utils as utils

# Add the parent directory of the `datasets` folder to the Python path
# Adjust the path below if your folder structure is different
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'diffuser')))

# Import load_environment
from diffuser.datasets.sequence import PrecomputedDualTrajectoryDataset

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('diffusion')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)
# # Test loading the environment
# if __name__ == "__main__":
#     try:
#         # Load maze2d-large-v1
#         env = load_environment("maze2d-large-v1")
#         print(f"Environment loaded successfully: {env}")
#     except Exception as e:
#         print(f"Failed to load environment: {e}")

# # Access the dataset from the environment
# dataset = env.get_dataset()

# # Print dataset keys to see available fields
# print(f"Dataset keys: {list(dataset.keys())}")
# print(env._target)

# # Check the size of observations, actions, rewards, etc.
# for key, value in dataset.items():
#     if isinstance(value, np.ndarray):
#         print(f"{key}: shape={value.shape}, dtype={value.dtype}")

# import matplotlib.pyplot as plt

# # Access the maze array
# maze_array = env.maze_arr  # Binary array where 1 = wall, 0 = free space

# # Plot the maze
# plt.imshow(maze_array, cmap="gray")
# plt.title("Maze2D-Large Layout")
# plt.savefig('g.png')

import numpy as np
import os
from tqdm import tqdm
import torch

def precompute_dual_trajectory_dataset(single_agent_dataset, output_file, num_samples=10000):
    """
    Precompute a dataset combining two single-agent trajectories into one sample.

    Args:
        single_agent_dataset: An instance of SequenceDataset with single-agent trajectories.
        output_file: Path to save the precomputed dataset.
        num_samples: Number of dual-trajectory samples to generate.
    """
    trajectory1_list = []
    trajectory2_list = []
    conditions_list = []

    print("Generating dual-trajectory dataset...")
    for _ in tqdm(range(num_samples)):
        # Randomly select two trajectories
        idx1, idx2 = np.random.choice(len(single_agent_dataset), size=2, replace=False)
        trajectory1 = single_agent_dataset[idx1].trajectories
        trajectory2 = single_agent_dataset[idx2].trajectories

        # Ensure the start and goal points are distinct
        start1, goal1 = trajectory1[0][single_agent_dataset.action_dim:], trajectory1[-1][single_agent_dataset.action_dim:]
        start2, goal2 = trajectory2[0][single_agent_dataset.action_dim:], trajectory2[-1][single_agent_dataset.action_dim:]
        # if np.allclose(start1, start2) or np.allclose(goal1, goal2):
        #     continue  # Skip if start or goal points overlap

        # Maximize the distance between the two trajectories
        # trajectory2 = maximize_distance(trajectory1, trajectory2)

        # Save the trajectories and conditions
        trajectory1_list.append(trajectory1)
        trajectory2_list.append(trajectory2)
        conditions_list.append({
            "start1": start1,
            "goal1": goal1,
            "start2": start2,
            "goal2": goal2,
        })

    # Save the precomputed dataset
    np.savez_compressed(output_file,
                        trajectory1=np.array(trajectory1_list),
                        trajectory2=np.array(trajectory2_list),
                        conditions=np.array(conditions_list))
    print(f"Dual-trajectory dataset saved to {output_file}")


def maximize_distance(trajectory1, trajectory2):
    """
    Modify trajectory2 to maximize its distance from trajectory1.
    """
    for t in range(len(trajectory1)):
        point1 = trajectory1[t, :2]  # (x, y) of trajectory1
        point2 = trajectory2[t, :2]  # (x, y) of trajectory2

        # Push point2 away from point1
        delta = point2 - point1
        norm_delta = delta / (np.linalg.norm(delta) + 1e-5)
        trajectory2[t, :2] += 0.1 * norm_delta  # Move away by a small step

    return trajectory2


if __name__ == "__main__":
    from diffuser.datasets.sequence import SequenceDataset

    # Load the single-agent dataset
    single_agent_dataset = dataset_config()

    # Precompute the dual-trajectory dataset
    output_file = "dual_trajectory_dataset_10k.npz"
    precompute_dual_trajectory_dataset(single_agent_dataset, output_file, num_samples=10000)
    # precompute_dual_trajectory_dataset(single_agent_dataset, output_file, num_samples=len(single_agent_dataset))

    # Load the precomputed dataset for verification
    dual_dataset = PrecomputedDualTrajectoryDataset(output_file, single_agent_dataset.action_dim, single_agent_dataset.observation_dim)
    print(f"Loaded precomputed dataset with {len(dual_dataset)} samples.")



