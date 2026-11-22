#! /bin/bash

# The commented-out part is used in SLURM environments 
# #SBATCH --job-name=2dU1CNNrun_001
# #SBATCH --output=logs/slurm/u1dm_%A_%a.out
# #SBATCH --error=logs/slurm/u1dm_%A_%a.err

# #SBATCH --ntasks=1        # Number of nodes

# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=1
# #SBATCH --partition=gpu    # Specify the partition
# #SBATCH --mem=2G
# #SBATCH --account=<project_id>
# #SBATCH --array=0-7

# module load anaconda/2024.06
# source activate workenv

CONFIG_FILE=configs/train_config.yaml

SAMPLE_SIZE=1_000
LATTICE_SIZE=16
CHANNELS=2

# Cannot do float math in bash; 
# TRAIN_BETA=2.000
TARGET_BETA=6.000
RESCALING_FACTOR=3.000

SHAPE="[${SAMPLE_SIZE}, ${CHANNELS}, ${LATTICE_SIZE}, ${LATTICE_SIZE}]"

L_STEPS=(10 25) 
NUM_STEPS=(2000 5000) 
echo "Starting sampling loop..."

# num_steps=${NUM_STEPS[$SLURM_ARRAY_TASK_ID]} 
num_steps=${NUM_STEPS[0]} 

# l_steps=${L_STEPS[$SLURM_ARRAY_TASK_ID]} 
l_steps=${L_STEPS[0]} 

for num_steps in "${NUM_STEPS[@]}"
do
    for sampler_name in maala euler heun poorman
    do
        for schedule_type in log
        do
            SAVEFILE=${sampler_name}_beta_${TARGET_BETA}_L${LATTICE_SIZE}_N${num_steps}_${schedule_type}.npy

            echo "Generating data using ${sampler_name} ${schedule_type} ${num_steps} steps"

            python src/generate_dm_data.py \
            --config "${CONFIG_FILE}" \
            --set sampler.name="${sampler_name}" \
            --set sampler.params.shape="${SHAPE}" \
            --set sampler.params.l_steps="${l_steps}" \
            --set sampler.params.num_steps="${num_steps}" \
            --set sampler.params.S_churn="$((2*num_steps))" \
            --set sampler.params.schedule_type="${schedule_type}" \
            --set sampler.params.rescaling_factor="${RESCALING_FACTOR}" \
            --savefile "${SAVEFILE}"
        done
    done
done

echo "Done!"