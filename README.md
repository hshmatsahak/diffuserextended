# CSC2626 Final Project - Extending the Diffuser Framework

This project builds upon the "Planning with Diffusion for Flexible Behavior Synthesis" framework, extending its capabilities to address key challenges in trajectory planning for goal-conditioned reinforcement learning (RL). Our contributions include:

- **Middle Waypoint Conditioning:** Incorporating intermediate waypoints to ensure globally coherent trajectories in complex environments.
- **Multi-Agent Trajectory Planning:** Introducing safe, coordinated planning for multiple agents with collision-avoidance mechanisms.
- **Dynamic Obstacle Avoidance:** Adapting trajectories in real-time to varying obstacle configurations.

These extensions were evaluated in the Maze2D task, demonstrating improvements in trajectory feasibility, safety, and planning efficiency. This work advances the state-of-the-art in diffusion-based planning, making it applicable to real-world scenarios in robotics, autonomous systems, and collaborative environments.

<p align="center">
    <img src="https://diffusion-planning.github.io/images/diffuser-card.png" width="60%" title="Diffuser model">
</p>

---

## Quickstart

Train the diffusion model using the following command:
```bash
python scripts/train.py --exp_name <exp_name>
```

Where `<exp_name>` must be one of the following:
- `maze2d`
- `maze2d_midgoal`
- `maze2d_condnet`
- `maze2d_dual`
- `maze2d_condnet_obstacle`

The outputs will be saved in the `logs/` directory.

---

## Installation

1. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate diffusion
   ```

2. Install the project:
   ```bash
   pip install -e .
   ```

---

## Usage

### Training
Train a diffusion model with:
```bash
python scripts/train.py --exp_name maze2d
```

### Planning
Generate plans using the trained diffusion model with:
```bash
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1
```

---

## Acknowledgements

This work is based on the code from "Planning with Diffusion for Flexible Behavior Synthesis" by Michael Janner, Yilun Du, Joshua B. Tenenbaum, and Sergey Levine.
