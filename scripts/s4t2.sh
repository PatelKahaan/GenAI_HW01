#!/bin/bash

history_lengths=(20 40 60 80 100 120 140)

submit_job() {
    local history_length=$1
    experiment_name="rnn_wo_att_seq_${history_length}"

    echo "Submitting experiment: ${experiment_name} with sequence length ${history_length}..."

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

cd "/share/csc591s25/kpatel48/GenAI-for-Systems-Gym/homework-1/models/RNN_without_Attention"

python3 -m cache_replacement.policy_learning.cache_model.main \
    --experiment_base_dir=/share/csc591s25/kpatel48/tmp \
    --experiment_name=${experiment_name} \
    --cache_configs="cache_replacement/policy_learning/cache/configs/default.json" \
    --model_configs="cache_replacement/policy_learning/cache_model/configs/default.json" \
    --model_bindings="loss=[\"ndcg\", \"reuse_dist\"]" \
    --model_bindings="address_embedder.max_vocab_size=5000" \
    --model_bindings="sequence_length=${history_length}" \
    --train_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_train.csv" \
    --valid_memtrace="cache_replacement/policy_learning/cache/traces/astar_313B_valid.csv" \
    --total_steps=20001
EOF
}

for history_length in "${history_lengths[@]}"; do
    submit_job "$history_length" &
done

echo "All RNN Without Attention history length experiments submitted successfully!"
