#!/bin/bash

learning_rates=(0.00001 0.001 0.1)

submit_job() {
    local learning_rate=$1
    experiment_name="mlp_lr_${learning_rate}"

    echo "Submitting experiment: ${experiment_name} with learning rate ${learning_rate}..."

    bsub <<EOF
#!/bin/bash
#BSUB -n 1
#BSUB -W 24:00
#BSUB -q gpu
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -J ${experiment_name}
#BSUB -o /share/csc591s25/kpatel48/logs/${experiment_name}.out.%J
#BSUB -e /share/csc591s25/kpatel48/logs/${experiment_name}.err.%J

source ~/.bashrc
conda activate /share/csc591s25/conda_env/new_env

cd "/share/csc591s25/kpatel48/GenAI-for-Systems-Gym/homework-1/models/MLP"

python3 -m cache_replacement.policy_learning.cache_model.main \
    --experiment_base_dir=/share/csc591s25/kpatel48/tmp \
    --experiment_name=${experiment_name} \
    --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
    --model_configs="cache_replacement/policy_learning/cache_model/configs/default.json" \
    --model_bindings="lr=${learning_rate}" \
    --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_train.csv" \
    --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv" \
    --total_steps=20001
EOF
}

for learning_rate in "${learning_rates[@]}"; do
    submit_job "$learning_rate" &
done

echo "All learning rate experiments submitted successfully!"