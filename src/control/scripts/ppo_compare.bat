@echo off
SETLOCAL EnableDelayedExpansion

REM Set timestamp for unique run folder
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"
set "runs_dir=runs_comparison_%timestamp%"

echo Creating runs directory: %runs_dir%
mkdir %runs_dir%

echo ======================================================
echo Starting PPO-CLIP training...
echo ======================================================

REM Run PPOCLIP with optimal parameters
python train_ppo.py ^
  --algorithm PPOCLIP ^
  --env-name VectorizedDD ^
  --obstacle-shape square ^
  --seed 123456 ^
  --num-envs 32 ^
  --gamma 0.99 ^
  --tau 0.95 ^
  --hidden-size 256 ^
  --batch-size 8192 ^
  --num-steps 5000000 ^
  --update-interval 2048 ^
  --num-mini-batch 4 ^
  --clip-param 0.1 ^
  --policy-lr 2.5e-4 ^
  --value-lr 5e-4 ^
  --ppo-epoch 4 ^
  --entropy-coef 0.01 ^
  --value-loss-coef 0.5 ^
  --max-grad-norm 0.5 ^
  --use-clipped-value-loss ^
  --eval-interval 5 ^
  --eval-episodes 10 ^
  --experiment-name "PPOCLIP_Comparison_%timestamp%" ^
  --cuda ^
  --save-curve

echo ======================================================
echo Starting PPO-KL training...
echo ======================================================

REM Run PPOKL with optimal parameters
python train_ppo.py ^
  --algorithm PPOKL ^
  --env-name VectorizedDD ^
  --obstacle-shape square ^
  --seed 123456 ^
  --num-envs 32 ^
  --gamma 0.995 ^
  --tau 0.95 ^
  --hidden-size 256 ^
  --batch-size 8192 ^
  --num-steps 5000000 ^
  --update-interval 2048 ^
  --num-mini-batch 4 ^
  --policy-lr 1e-4 ^
  --value-lr 3e-4 ^
  --ppo-epoch 5 ^
  --entropy-coef 0.003 ^
  --value-loss-coef 0.5 ^
  --max-grad-norm 0.5 ^
  --kl-target 0.005 ^
  --kl-beta 1.0 ^
  --kl-adaptive ^
  --kl-cutoff-factor 4.0 ^
  --kl-cutoff-coef 2000.0 ^
  --min-kl-coef 0.02 ^
  --max-kl-coef 5.0 ^
  --eval-interval 5 ^
  --eval-episodes 10 ^
  --experiment-name "PPOKL_Comparison_%timestamp%" ^
  --cuda ^
  --save-curve

echo ======================================================
echo Running PPO analysis on results...
echo ======================================================

REM Move runs to the comparison directory
mkdir %runs_dir%\runs
move runs\* %runs_dir%\runs\

REM Run analysis script on the collected data
python ppo_analysis.py ^
  --logs-dir %runs_dir%\runs ^
  --output-dir %runs_dir%\analysis ^
  --smoothing 10 ^
  --figsize 15,10 ^
  --dpi 300 ^
  --font-size 12

echo ======================================================
echo Analysis complete!
echo ======================================================
echo Results are available in: %runs_dir%\analysis
echo Learning curves and statistical analysis have been generated.

pause