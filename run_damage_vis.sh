#!/bin/bash

# Define the directory containing the .pkl files
DIR="./rollouts/Fragment/Step-0-100-3-AllTest/"

# Loop through each .pkl file in the directory
for FILE in "$DIR"/*.pkl; do
    # Extract the rollout_name from the filename
    ROLLOUT_NAME=$(basename "$FILE" .pkl)

    # Run the Python command with the extracted rollout_name
    python -m gns.render_fragment --rollout_dir="$DIR" --rollout_name="$ROLLOUT_NAME" --step_stride=1
done
