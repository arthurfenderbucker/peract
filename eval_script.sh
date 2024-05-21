
export PERACT_ROOT=/home/arthur/Desktop/CMU/research/benchmarks/peract
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 LIBGL_DEBUG=verbose CUDA_VISIBLE_DEVICES=0 python eval.py     rlbench.tasks=[open_drawer]     rlbench.task_name='multi'     rlbench.demo_path=$PERACT_ROOT/data/val     framework.gpu=0     framework.logdir=$PERACT_ROOT/ckpts/     framework.start_seed=0     framework.eval_envs=1     framework.eval_from_eps_number=0     framework.eval_episodes=10     framework.csv_logging=True     framework.tensorboard_logging=True     framework.eval_type='last'     rlbench.headless=False
