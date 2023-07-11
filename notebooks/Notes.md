# Cmd to run
# Train
CUDA_VISIBLE_DEVICES=0 python -m gns.train --mode=train --data_path=/home/jovyan/work/data_temp/fragment/Fragment/ --model_path=./models/Fragment/ --output_path=./rollouts/Fragment/ --batch_size=1 --noise_std=0.05 --connection_radius=20 --layers=5 --hidden_dim=32 --lr_init=0.001 --ntraining_steps=500000 --lr_decay_steps=150000 --dim=3 --project_name=Fragment-3D --run_name=ns0.05_R11_L5N32 --nsave_steps=5000 --log=False

CUDA_VISIBLE_DEVICES=7 python -m gns.train --mode=train --data_path=./data/Concrete3D/ --model_path=./models/Concrete3D/ --output_path=./rollouts/Concrete3D/ --batch_size=1 --noise_std=0.001 --connection_radius=15 --layers=10 --lr_init=0.001 --ntraining_steps=300000 --lr_decay_steps=80000 --dim=3d --project_name=GNS-3D --run_name=ns1e-3 --log=False

# Rollout
CUDA_VISIBLE_DEVICES=6 python -m gns.train -mode=rollout --data_path=/home/jovyan/work/data_temp/fragment/Fragment/ --model_path=./models/Fragment/ns0.1_R17_L5N64_strength/ --model_file=model-039000.pt --output_path=./rollouts/Fragment/ --noise_std=0.1 --dim=3 --connection_radius=17 --layers=5 --hidden_dim=64

CUDA_VISIBLE_DEVICES=5 python -m gns.train -mode=rollout --data_path=./data/Concrete2D-C-mps/ --model_path=./models/Concrete2D-C-mps/ --model_file=mps_bs2_ns6.7e-5_R0.01_I1M_proceassed-model-307000.pt --output_path=./rollouts/Concrete2D-C/ --noise_std=0.000067 --dim=2d --connection_radius=0.01

# Visualisation
python -m gns.render_rollout_3d --rollout_path=rollouts/Fragment/rollout_0.pkl --output_path=rollouts/Fragment/rollout_0.jif


# Noise scale
Noise scale should be propotional to the velocity scale
- 2D-C: vel std (3.1e-5, 4.7e-5) ==> noise std (6.7e-5, 6.7e-5)
- 2D-I: vel std (4.7e-5, 8.6e-5) ==> noise std (5e-5, 5e-5)
- 2D-T:  vel std (5.6e-3, 3.4e-3) ==> noise std (3e-4, 3e-4)
- 3D-Fragment: vel std (0.1, 0.1, 0.8) ==> noise std (?, ?, ?)

# Notes
- wandb step increase by default everytime wandb.log() is called
- For quasi-static simulation, many particles have no acceleartion in many timesteps. Hence, the sampled training steps might have many zero ground truth or not, resulting in
    a large difference between training iterations, as shown by the training loss. This might be the reason that the training loss stucks quickly at some point
- Adding noise significantly decreases the training loss but the GNN is probably fitting the Gaussian noise. This is evidenced by the relative constant rollout (all particles move
    the same as the learning is on noise) 
