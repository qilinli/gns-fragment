#!/bin/bash

# Fail on any error.
set -e

# # Display commands being run.
set -x

for i in $(seq 0 1 14)
do
python -m gns.render_rollout_3d --rollout_name=rollout_$i --output_postfix='back'
done

