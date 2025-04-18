@echo off
setlocal enabledelayedexpansion

REM Default parameters
set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=VectorizedDD

set PPO_STEPS=%2
if "%PPO_STEPS%"=="" set PPO_STEPS=5000000

set SAC_STEPS=%3
if "%SAC_STEPS%"=="" set SAC_STEPS=500000

set NUM_ENVS=%4
if "%NUM_ENVS%"=="" set NUM_ENVS=4

set OBSTACLE_SHAPE=%5
if "%OBSTACLE_SHAPE%"=="" set OBSTACLE_SHAPE=square

set PPO_ALGORITHM=%6
if "%PPO_ALGORITHM%"=="" set PPO_ALGORITHM=PPOCLIP

set ENTROPY_MODE=%7
if "%ENTROPY_MODE%"=="" set ENTROPY_MODE=adaptive

REM Create timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

REM Setup directories
set RUN_NAME=%ENV_NAME%_%PPO_ALGORITHM%_%PPO_STEPS%_SAC_%ENTROPY_MODE%_%SAC_STEPS%_%TIMESTAMP%
set OUTPUT_DIR=results\%RUN_NAME%

REM Create output directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

echo ===============================================
echo Starting RL Algorithm Comparison Experiment
echo ===============================================
echo Environment: %ENV_NAME%
echo Obstacle Shape: %OBSTACLE_SHAPE%
echo PPO Steps: %PPO_STEPS%
echo SAC Steps: %SAC_STEPS%
echo Parallel environments: %NUM_ENVS%
echo Output directory: %OUTPUT_DIR%
echo ===============================================

REM Run the comparison script
echo Running algorithm comparison...
python compare_agents.py ^
    --env-name "%ENV_NAME%" ^
    --obstacle-shape "%OBSTACLE_SHAPE%" ^
    --num-steps "%SAC_STEPS%" ^
    --ppo-num-steps "%PPO_STEPS%" ^
    --num-envs "%NUM_ENVS%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --run-name "%RUN_NAME%" ^
    --ppo-algorithm "%PPO_ALGORITHM%" ^
    --entropy-mode "%ENTROPY_MODE%" ^
    --use-twin-critic ^
    --eval-interval 5000 ^
    --ppo-eval-interval 50000 ^
    --log-interval 1000 ^
    --checkpoint-interval 20000 ^
    --ppo-checkpoint-interval 200000 ^
    --lr-annealing ^
    --normalize-advantages ^
    --verbose

if %ERRORLEVEL% NEQ 0 (
    echo Error: Algorithm comparison failed. Check logs for details.
    exit /b 1
)

REM Run analysis script
echo Running analysis on results...
python analysis.py ^
    --results-dir "%OUTPUT_DIR%" ^
    --smoothing 3 ^
    --dpi 300 ^
    --normalize-steps

if %ERRORLEVEL% NEQ 0 (
    echo Error: Analysis failed. Check logs for details.
    exit /b 1
)

echo ===============================================
echo Experiment and analysis completed successfully!
echo Results saved to: %OUTPUT_DIR%
echo To view TensorBoard logs, run:
echo tensorboard --logdir=%OUTPUT_DIR%\tensorboard
echo ===============================================

endlocal