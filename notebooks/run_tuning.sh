#!/bin/bash
# Copyright 2020 Deepmind Technologies Limited.
# Copyright 2021 Geoelements.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fail on any error.
set -e

# Display commands being run.
set -x

CUDA_VISIBLE_DEVICES=2 python -m gns.train --mode=train --data_path=./data/Concrete1D/ --model_path=./models/Concrete1D/ --output_path=./rollouts/Concrete1D/ --batch_size=4 --noise_std=0.001 --connection_radius=0.025 --layers=10 --lr_init=0.001 --ntraining_steps=500000 --lr_decay_steps=200000 --dim=1d --project_name=GNS-1D-strain --run_name=BS4_ns1e-3_R0.025 --log=True

CUDA_VISIBLE_DEVICES=2 python -m gns.train --mode=train --data_path=./data/Concrete1D/ --model_path=./models/Concrete1D/ --output_path=./rollouts/Concrete1D/ --batch_size=8 --noise_std=0.001 --connection_radius=0.03 --layers=10 --lr_init=0.001 --ntraining_steps=500000 --lr_decay_steps=200000 --dim=1d --project_name=GNS-1D-strain --run_name=BS8_ns1e-3_R0.03 --log=True
CUDA_VISIBLE_DEVICES=2 python -m gns.train --mode=train --data_path=./data/Concrete1D/ --model_path=./models/Concrete1D/ --output_path=./rollouts/Concrete1D/ --batch_size=4 --noise_std=0.001 --connection_radius=0.035 --layers=10 --lr_init=0.001 --ntraining_steps=500000 --lr_decay_steps=200000 --dim=1d --project_name=GNS-1D-strain --run_name=BS4_ns1e-3_R0.035 --log=True


