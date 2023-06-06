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

# # Display commands being run.
set -x

for i in $(seq 0 1 9)
do
python -m gns.render_rollout_$1 --rollout_path=rollouts/$2/rollout_$i.pkl --output_path=rollouts/$2/rollout_$i.gif
done