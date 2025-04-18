#!/bin/bash

# Script to run RL algorithm comparison experiments and analysis
# Usage: ./run_experiments.sh [env_name] [ppo_steps] [sac_steps] [num_envs] [obstacle_shape] [ppo_algorithm] [entropy_mode]

# Default parameters
ENV_NAME=${1:-"VectorizedDD"}            # Default to VectorizedDD environment
PPO_STEPS=${2:-5000000}                  # Default to 5M steps for PPO
SAC_STEPS=${3:-500000}                   # Default to 500K steps for SAC
NUM_ENVS=${4:-4}                         # Default to 4 parallel environments
OBSTACLE_SHAPE=${5:-"square"}            # Default to square obstacles
PPO_ALGORITHM=${6:-"PPOCLIP"}            # Default to PPOCLIP (alternatives: PPOKL)
ENTROPY_MODE=${7:-"adaptive"}            # Default to adaptive entropy (alternatives: fixed, none)

# Calculate evaluation intervals based on steps
# For PPO, evaluate approximately every 1% of total steps
PPO_EVAL_INTERVAL=$((PPO_STEPS / 100))
# For SAC, evaluate approximately every 2% of total steps
SAC_EVAL_INTERVAL=$((SAC_STEPS / 50))

# Calculate checkpoint intervals - save approx. 25 checkpoints during training
PPO_CHECKPOINT_INTERVAL=$((PPO_STEPS / 25))
SAC_CHECKPOINT_INTERVAL=$((SAC_STEPS / 25))

# Ensure intervals are reasonable (not too small)
if [ $PPO_EVAL_INTERVAL -lt 10000 ]; then
    PPO_EVAL_INTERVAL=10000
fi
if [ $SAC_EVAL_INTERVAL -lt 5000 ]; then
    SAC_EVAL_INTERVAL=5000
fi
if [ $PPO_CHECKPOINT_INTERVAL -lt 50000 ]; then
    PPO_CHECKPOINT_INTERVAL=50000
fi
if [ $SAC_CHECKPOINT_INTERVAL -lt 10000 ]; then
    SAC_CHECKPOINT_INTERVAL=10000
fi

# Set up experiment name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${ENV_NAME}_${PPO_ALGORITHM}_${PPO_STEPS}_SAC_${ENTROPY_MODE}_${SAC_STEPS}_${TIMESTAMP}"
OUTPUT_DIR="results/${RUN_NAME}"

# Check if required python modules are available
python -c "import torch, gymnasium, tqdm, matplotlib.pyplot, seaborn, pandas" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Missing required Python modules. Please install using:"
    echo "pip install torch gymnasium tqdm matplotlib seaborn pandas"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "==============================================="
echo "Starting RL Algorithm Comparison Experiment"
echo "==============================================="
echo "Environment: $ENV_NAME"
echo "Obstacle Shape: $OBSTACLE_SHAPE"
echo "PPO Algorithm: $PPO_ALGORITHM - $PPO_STEPS steps"
echo "SAC Configuration: $SAC_STEPS steps"
echo "  - Entropy Mode: $ENTROPY_MODE"
echo "Parallel environments: $NUM_ENVS"
echo "Output directory: $OUTPUT_DIR"
echo "==============================================="

# Run the comparison script with all parameters
echo "Running algorithm comparison..."
python compare_agents.py \
    --env-name "$ENV_NAME" \
    --obstacle-shape "$OBSTACLE_SHAPE" \
    --num-steps "$SAC_STEPS" \
    --ppo-num-steps "$PPO_STEPS" \
    --num-envs "$NUM_ENVS" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    --ppo-algorithm "$PPO_ALGORITHM" \
    --entropy-mode "$ENTROPY_MODE" \
    --use-twin-critic \
    --eval-interval "$SAC_EVAL_INTERVAL" \
    --ppo-eval-interval "$PPO_EVAL_INTERVAL" \
    --checkpoint-interval "$SAC_CHECKPOINT_INTERVAL" \
    --ppo-checkpoint-interval "$PPO_CHECKPOINT_INTERVAL" \
    --log-interval 1000 \
    --lr-annealing \
    --normalize-advantages \
    --verbose

# Check if the comparison completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Algorithm comparison failed. Check logs for details."
    exit 1
fi

# Run analysis script (using our updated version that handles different step counts)
echo "Running analysis on results..."
python analysis.py \
    --results-dir "$OUTPUT_DIR" \
    --smoothing 3 \
    --dpi 300 \
    --normalize-steps

# Check if analysis completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Analysis failed. Check logs for details."
    exit 1
fi

echo "==============================================="
echo "Experiment and analysis completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo "To view TensorBoard logs, run:"
echo "tensorboard --logdir=$OUTPUT_DIR/tensorboard"
echo "==============================================="

# Create a PDF report using the analysis results
if command -v convert &> /dev/null; then
    echo "Creating PDF report..."
    
    # Directory for analysis plots
    ANALYSIS_DIR="$OUTPUT_DIR/analysis"
    
    # Check if analysis directory exists
    if [ -d "$ANALYSIS_DIR" ]; then
        # Combine plots into a PDF report
        convert $(find "$ANALYSIS_DIR" -name "*.png" | sort) "$OUTPUT_DIR/report.pdf"
        echo "PDF report created: $OUTPUT_DIR/report.pdf"
    else
        echo "Analysis directory not found. Skipping PDF report creation."
    fi
else
    echo "ImageMagick not found. Install it to generate PDF reports."
fi

# Run a quick evaluation of the trained models
echo "==============================================="
echo "Running evaluation of trained models..."
echo "==============================================="

# PPO evaluation
PPO_BEST_MODEL=$(find "$OUTPUT_DIR" -name "*ppo*best.pt" | head -n 1)
if [ -n "$PPO_BEST_MODEL" ]; then
    echo "Evaluating best PPO model: $PPO_BEST_MODEL"
    python evaluate_ppo.py \
        --env-name "$ENV_NAME" \
        --obstacle-shape "$OBSTACLE_SHAPE" \
        --algorithm "$PPO_ALGORITHM" \
        --checkpoint "$PPO_BEST_MODEL" \
        --num-episodes 10 \
        --render
fi

# SAC evaluation
SAC_BEST_MODEL=$(find "$OUTPUT_DIR" -name "*sac*best.pt" | head -n 1)
if [ -n "$SAC_BEST_MODEL" ]; then
    echo "Evaluating best SAC model: $SAC_BEST_MODEL"
    python evaluate_sac.py \
        --env-name "$ENV_NAME" \
        --obstacle-shape "$OBSTACLE_SHAPE" \
        --checkpoint "$SAC_BEST_MODEL" \
        --num-episodes 10 \
        --render
fi

echo "==============================================="
echo "Experiment complete!"
echo "==============================================="