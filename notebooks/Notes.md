# Cmd to run
# Train
CUDA_VISIBLE_DEVICES=7 python -m gns.train --mode=train --data_path=/home/jovyan/share/8TB-share/qilin/fragment/Step-0-100-3/ --model_path=./models/Fragment/ --output_path=./rollouts/Fragment/ --batch_size=1 --noise_std=0.0005 --connection_radius=14 --layers=5 --hidden_dim=64 --lr_init=0.001 --ntraining_steps=200001 --dim=3 --project_name=Fragment-3D-full --run_name=NS5e-4xy_1e-2z_R14_L5N64_PosNsx10_LRCosine --nsave_steps=1000 --log=1

CUDA_VISIBLE_DEVICES=7 python -m gns.train --mode=train --data_path=./data/Concrete3D/ --model_path=./models/Concrete3D/ --output_path=./rollouts/Concrete3D/ --batch_size=1 --noise_std=0.001 --connection_radius=15 --layers=10 --lr_init=0.001 --ntraining_steps=300000 --lr_decay_steps=80000 --dim=3d --project_name=GNS-3D --run_name=ns1e-3 --log=False

# Rollout
CUDA_VISIBLE_DEVICES=5 python -m gns.train -mode=rollout --data_path=/home/jovyan/share/8TB-share/qilin/fragment/inference/ --model_path=./models/Fragment/Benchmark-NS5e-4_1e-2_R14_L5N64_PosNsx10_LRCosine/ --model_file=model-097000.pt --output_path=./rollouts/Fragment/inference/ --noise_std=0.0005 --dim=3 --connection_radius=14 --layers=5 --hidden_dim=64

CUDA_VISIBLE_DEVICES=5 python -m gns.train -mode=rollout --data_path=/home/jovyan/share/8TB-share/qilin/fragment/Step-0-100-3-All-In-Test/ --model_path=./models/Fragment/Benchmark-NS5e-4_1e-2_R14_L5N64_PosNsx10_LRCosine/ --model_file=model-097000.pt --output_path=./rollouts/Fragment/temp/ --noise_std=0.0005 --dim=3 --connection_radius=14 --layers=5 --hidden_dim=64

# Visualisation
python -m gns.render_rollout_3d --rollout_dir=rollouts/Fragment/inference/ --rollout_name=d3plot --step_stride=3


# Noise scale
Noise scale should be propotional to the velocity scale
- 2D-C:        vel std (3.1e-5, 4.7e-5),         acc std (2.1e-6, 3.4e-6) ==>         noise std (6.7e-5, 6.7e-5)
- 2D-I:        vel std (4.7e-5, 8.6e-5),         acc std (4.7e-6, 9.9e-6) ==>         noise std (5e-5, 5e-5)
- 2D-T:        vel std (5.6e-3, 3.4e-3),         acc std (4.4e-4, 3.8e-4) ==>         noise std (3e-4, 3e-4)
- 3D-Fragment: vel std (7.9e-2, 8.4e-2, 5.4e-1), acc std (2.3e-2, 3.7e-2, 2.3e-2) ==> noise std (5e-4)

# Notes
- wandb step increase by default everytime wandb.log() is called
- For quasi-static simulation, many particles have no acceleartion in many timesteps. Hence, the sampled training steps might have many zero ground truth or not, resulting in
    a large difference between training iterations, as shown by the training loss. This might be the reason that the training loss stucks quickly at some point
- Adding noise significantly decreases the training loss but the GNN is probably fitting the Gaussian noise. This is evidenced by the relative constant rollout (all particles move
    the same as the learning is on noise) 
