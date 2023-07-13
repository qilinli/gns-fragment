# Cmd to run
# Train
CUDA_VISIBLE_DEVICES=5 python -m gns.train --mode=train --data_path=/home/jovyan/share/8TB-share/qilin/fragment/060-Step-10-80-2/ --model_path=./models/Fragment/ --output_path=./rollouts/Fragment/ --batch_size=1 --noise_std=0.05 --connection_radius=17 --layers=5 --hidden_dim=56 --lr_init=0.001 --ntraining_steps=50000 --lr_decay_steps=15000 --dim=3 --project_name=Fragment-3D --run_name=NS10_R17_L5N56_History5_Step35_NewData --nsave_steps=10 --log=0

CUDA_VISIBLE_DEVICES=7 python -m gns.train --mode=train --data_path=./data/Concrete3D/ --model_path=./models/Concrete3D/ --output_path=./rollouts/Concrete3D/ --batch_size=1 --noise_std=0.001 --connection_radius=15 --layers=10 --lr_init=0.001 --ntraining_steps=300000 --lr_decay_steps=80000 --dim=3d --project_name=GNS-3D --run_name=ns1e-3 --log=False

# Rollout
CUDA_VISIBLE_DEVICES=6 python -m gns.train -mode=rollout --data_path=/home/jovyan/work/data_temp/fragment/Fragment/ --model_path=./models/Fragment/ns0.1_R17_L5N64_strength/ --model_file=model-039000.pt --output_path=./rollouts/Fragment/ --noise_std=0.1 --dim=3 --connection_radius=17 --layers=5 --hidden_dim=64

CUDA_VISIBLE_DEVICES=5 python -m gns.train -mode=rollout --data_path=/home/jovyan/work/data_temp/fragment/Fragment/ --model_path=./models/Fragment/ns0.01-0.08_R17_L5N56_History5_Step27/ --model_file=model-011000.pt --output_path=./rollouts/Fragment --connection_radius=17 --layers=5 --hidden_dim=56

# Visualisation
python -m gns.render_rollout_3d --rollout_path=rollouts/Fragment/rollout_0.pkl --output_path=rollouts/Fragment/rollout_0.jif


# Noise scale
Noise scale should be propotional to the velocity scale
- 2D-C:        vel std (3.1e-5, 4.7e-5),         acc std (2.1e-6, 3.4e-6) ==>         noise std (6.7e-5, 6.7e-5)
- 2D-I:        vel std (4.7e-5, 8.6e-5),         acc std (4.7e-6, 9.9e-6) ==>         noise std (5e-5, 5e-5)
- 2D-T:        vel std (5.6e-3, 3.4e-3),         acc std (4.4e-4, 3.8e-4) ==>         noise std (3e-4, 3e-4)
- 3D-Fragment: vel std (7.9e-2, 8.4e-2, 5.4e-1), acc std (2.3e-2, 3.7e-2, 2.3e-2) ==> noise std (?, ?, ?)

# Notes
- wandb step increase by default everytime wandb.log() is called
- For quasi-static simulation, many particles have no acceleartion in many timesteps. Hence, the sampled training steps might have many zero ground truth or not, resulting in
    a large difference between training iterations, as shown by the training loss. This might be the reason that the training loss stucks quickly at some point
- Adding noise significantly decreases the training loss but the GNN is probably fitting the Gaussian noise. This is evidenced by the relative constant rollout (all particles move
    the same as the learning is on noise) 
