# Cmd to run
# Train
CUDA_VISIBLE_DEVICES=1 python -m gns.train --mode=train --data_path=./data/Concrete2D-T/ --model_path=./models/Concrete2D-T/ --output_path=./rollouts/Concrete2D-T/ --batch_size=2 --noise_std=0.001 --connection_radius=0.03 --layers=10 --lr_init=0.001 --ntraining_steps=100000 --lr_decay_steps=30000 --dim=2d --project_name=GNS-2D-T --run_name=ns1e-3 --log=False

CUDA_VISIBLE_DEVICES=7 python -m gns.train --mode=train --data_path=./data/Concrete3D/ --model_path=./models/Concrete3D/ --output_path=./rollouts/Concrete3D/ --batch_size=1 --noise_std=0.001 --connection_radius=15 --layers=10 --lr_init=0.001 --ntraining_steps=300000 --lr_decay_steps=80000 --dim=3d --project_name=GNS-3D --run_name=ns1e-3 --log=False
# Rollout
CUDA_VISIBLE_DEVICES=0 python -m gns.train -mode=rollout --data_path=./data/Concrete2D-C/ --model_path=./models/Concrete2D-C/ --model_file=noise6.7e-5_R0.01_bs2_lr1e-3_step500k-model-178000.pt --output_path=./rollouts/Concrete2D-C/ --noise_std=0.00067 --dim=2d --connection_radius=0.01

CUDA_VISIBLE_DEVICES=5 python -m gns.train -mode=rollout --data_path=./data/Concrete2D-C-mps/ --model_path=./models/Concrete2D-C-mps/ --model_file=mps_bs2_ns6.7e-5_R0.01_I1M_proceassed-model-307000.pt --output_path=./rollouts/Concrete2D-C/ --noise_std=0.000067 --dim=2d --connection_radius=0.01
# Visualisation
python -m gns.render_rollout_1d --rollout_path=rollouts/Concrete1D/rollout_0.pkl --output_path=rollouts/Concrete1D/rollout_0.gif

CUDA_VISIBLE_DEVICES=7 python -m gns.train -mode=rollout --data_path=./data/Concrete1D/ --model_path=./models/Concrete1D/ --model_file=noise6.7e-4_R0.04_bs2_lr1e-3_step500k-model-326000.pt --output_path=./rollouts/Concrete1D-new/ --noise_std=0.00067 --dim=1d --connection_radius=0.03

# Notes
- If net config changed before evaluation, load weights may fail
- If subtle config changed, evaluation may have low results
- Train loss (acc) and val loss (pos) are not comparable currently
- wandb step increase by default everytime wandb.log() is called
- nparticles_per_example is the trajectory length
- For quasi-static simulation, many particles have no acceleartion in many timesteps. Hence, the sampled training steps might have many zero ground truth or not, resulting in
    a large difference between training iterations, as shown by the training loss. This might be the reason that the training loss stucks quickly at some point
- Adding noise significantly decreases the training loss but the GNN is probably fitting the Gaussian noise. This is evidenced by the relative constant rollout (all particles move
    the same as the learning is on noise)
- pytorch geometric caps the knn in radius_graph to be <=32
- If the simulation domain is normalised to [0,1], then R=0.04 givens around 24 neighbors per particles (NN).
    - R=0.02, NN=5
    - R=0.025, NN=7
    - R=0.027, NN=7
    - R=0.03, NN=13
    - R=0.04, NN=25
    - R=0.05, NN=34
    - R=0.06, NN=50
    - R=0.08, NN=75
- The original domain is x (-165, 165) and y (-10, 85). Normalise it to (0,1) and (0,1) will change the w/h ratio. 
- Be careful with the simulation domain, the Bullet in impact loading has made y too large unnessarily



# Hyper-parameters 1D
- noise-level:       1e-3 (much better than 6.7e-4, 10e-7 vs 10e-6)
- batch-size:        4 (better than 2 and 8)
- connection-radius: 0.03 (0.02 is fine but less stable)
- GNN layers:        10 (5 with bs2 didn't work good but seems ok with bs4 bs8)




# Thing to do
## About the x/y unit ratio
- Current best result is obtained when normalise both them to [0,1], which means x/y ratio changed
- This is not idea as the particle size has been changed, and neighbour in x,y are not the same
- With R=0.015, 2D would have 3 NN on one side of x, and 1 NN one one side of y



                # Qilin, Compute one_step_position_mse during training
                ### ===========================================================
                #         ground_truth_positions = position[:, -1,:].squeeze()

                #         next_position = simulator.predict_positions(
                #             current_positions=position.to(device),
                #             nparticles_per_example=n_particles_per_example.to(device),
                #             particle_types=particle_type.to(device),
                #         )

                #         # Update kinematic particles from prescribed trajectory.
                #         kinematic_mask = (particle_type == KINEMATIC_PARTICLE_ID).clone().detach().to(device)
                #         kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, 2)
                #         next_position = torch.where(
                #             kinematic_mask, ground_truth_positions.to(device), next_position)

                #         one_step_position_mse = (next_position - ground_truth_positions.to(device)) ** 2
                #         # print(f"One_step_position_mse: {one_step_position_mse.mean(axis=0).sum()}")
                #         log["train/loss-one-step-position"] = one_step_position_mse.mean(axis=0).sum()
                ### ===========================================================