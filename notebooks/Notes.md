# Cmd to run
# Train
CUDA_VISIBLE_DEVICES=0 python -m gns.train --mode=train --data_path=/home/jovyan/work/data_temp/fragment/Fragment/ --model_path=./models/Fragment/ --output_path=./rollouts/Fragment/ --batch_size=1 --noise_std=0.05 --connection_radius=20 --layers=5 --hidden_dim=32 --lr_init=0.001 --ntraining_steps=500000 --lr_decay_steps=150000 --dim=3 --project_name=Fragment-3D --run_name=ns0.05_R11_L5N32 --nsave_steps=5000 --log=False

CUDA_VISIBLE_DEVICES=7 python -m gns.train --mode=train --data_path=./data/Concrete3D/ --model_path=./models/Concrete3D/ --output_path=./rollouts/Concrete3D/ --batch_size=1 --noise_std=0.001 --connection_radius=15 --layers=10 --lr_init=0.001 --ntraining_steps=300000 --lr_decay_steps=80000 --dim=3d --project_name=GNS-3D --run_name=ns1e-3 --log=False

# Rollout
CUDA_VISIBLE_DEVICES=6 python -m gns.train -mode=rollout --data_path=/home/jovyan/work/data_temp/fragment/Fragment/ --model_path=./models/Fragment/ns0.1_R17_L5N64_strength/ --model_file=model-039000.pt --output_path=./rollouts/Fragment/ --noise_std=0.1 --dim=3 --connection_radius=17 --layers=5 --hidden_dim=64

CUDA_VISIBLE_DEVICES=5 python -m gns.train -mode=rollout --data_path=./data/Concrete2D-C-mps/ --model_path=./models/Concrete2D-C-mps/ --model_file=mps_bs2_ns6.7e-5_R0.01_I1M_proceassed-model-307000.pt --output_path=./rollouts/Concrete2D-C/ --noise_std=0.000067 --dim=2d --connection_radius=0.01

# Visualisation
python -m gns.render_rollout_3d --rollout_path=rollouts/Fragment/rollout_0.pkl --output_path=rollouts/Fragment/rollout_0.jif


# Notes
- If net config changed before evaluation, load weights may fail
- If subtle config changed, evaluation may have low results
- Train loss (acc) and val loss (pos) are not comparable currently
- wandb step increase by default everytime wandb.log() is called
- For quasi-static simulation, many particles have no acceleartion in many timesteps. Hence, the sampled training steps might have many zero ground truth or not, resulting in
    a large difference between training iterations, as shown by the training loss. This might be the reason that the training loss stucks quickly at some point
- Adding noise significantly decreases the training loss but the GNN is probably fitting the Gaussian noise. This is evidenced by the relative constant rollout (all particles move
    the same as the learning is on noise)
- pytorch geometric caps the knn in radius_graph to be <=32
- The original domain is x (-165, 165) and y (-10, 85). Normalise it to (0,1) and (0,1) will change the w/h ratio. 
- Be careful with the simulation domain, the Bullet in impact loading has made y too large unnessarily
