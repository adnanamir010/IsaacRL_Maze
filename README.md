# Soft Actor-Critic (SAC) for Robot Navigation

This project implements the Soft Actor-Critic (SAC) reinforcement learning algorithm for robot navigation in simulated environments. The implementation features a custom 2D differential drive robot environment and supports both continuous and discrete action spaces.

## Features

- **SAC Algorithm**: Implementation of SAC with automatic entropy tuning
- **Custom Environment**: 2D navigation environment with obstacles
- **Vectorized Training**: Support for parallel training environments
- **Visualization**: Real-time rendering and progress tracking
- **Memory-Efficient**: Optimized for performance and memory usage
- **TensorBoard Integration**: Logging of training metrics

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/adnanamir010/IsaacRL_Maze.git
   cd IsaacRL_Maze/src/control/scripts
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `agents.py`: Implementation of the SAC agent
- `models.py`: Neural network architectures for policy and value functions
- `memory.py`: Replay buffer for off-policy learning
- `environment.py`: Custom robot navigation environment
- `train_sac.py`: Main training script
- `evaluate_sac.py`: Evaluation script for trained models
- `rl_utils.py`: Utility functions for reinforcement learning

## Training

To train a SAC agent on the custom navigation environment:

```bash
python train_sac.py --env-name VectorizedDD --num-envs 4 --batch-size 256 --hidden-size 128
```

### Key Parameters

- `--env-name`: Environment name (default: "VectorizedDD")
- `--num-envs`: Number of parallel environments (default: 4)
- `--batch-size`: Batch size for training (default: 256)
- `--hidden-size`: Hidden layer size for neural networks (default: 128)
- `--num-steps`: Maximum number of steps (default: 1,000,000)
- `--render`: Enable visualization (only for single environment)

For a full list of parameters, run:
```bash
python train_sac.py --help
```

## Evaluation

To evaluate a trained agent:

```bash
python evaluate_sac.py --env-name VectorizedDD --checkpoint checkpoints/sac_checkpoint_VectorizedDD_final_1000 --num-episodes 10 --render
```

### Key Parameters

- `--checkpoint`: Path to the checkpoint file
- `--num-episodes`: Number of evaluation episodes
- `--render`: Enable visualization

## Environment

The custom environment (`VectorizedDDEnv`) simulates a 2D differential drive robot navigating through obstacles to reach a goal. The environment features:

- Lidar-based observations
- Randomly generated obstacles
- Customizable rewards for goal-reaching, collision avoidance, and efficient navigation
- Real-time visualization with pygame

## TensorBoard Monitoring

You can monitor training progress using TensorBoard:

```bash
tensorboard --logdir runs/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Implementation based on the SAC paper by Haarnoja et al. (2018)
- Environment inspired by navigation challenges in robotics