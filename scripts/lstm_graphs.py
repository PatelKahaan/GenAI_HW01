import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tensorboard_base_dir = r"C:\Users\Kahaan Patel\Desktop\Sem II\GenAI for Comp Sys\HW01\tensorboard"
save_dir = r"C:\Users\Kahaan Patel\Desktop\Sem II\GenAI for Comp Sys\HW01\graphs"
os.makedirs(save_dir, exist_ok=True)

def extract_scalar_values(log_dir):
    event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not event_files:
        return None, None

    event_path = os.path.join(log_dir, event_files[0])
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()

    cache_hit_rate_tag = next((tag for tag in event_acc.Tags()["scalars"] if "cache_hit_rate" in tag), None)
    if not cache_hit_rate_tag:
        return None, None

    events = event_acc.Scalars(cache_hit_rate_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

experiments_embedder = {
    "LSTM Baseline": "LSTM_baseline",
    "Byte Embedder": "LSTM_byte_embedder",
}

experiments_reuse = {
    "LSTM Baseline": "LSTM_baseline",
    "Ablation - Reuse Distance Loss": "LSTM_abl_reuse",
}

experiments_ranking = {
    "LSTM Baseline": "LSTM_baseline",
    "Ablation - Ranking Loss": "LSTM_abl_rank",
}

def generate_plot(experiments, title, filename):
    plt.figure(figsize=(10, 5))
    
    for label, exp in experiments.items():
        log_dir = os.path.join(tensorboard_base_dir, exp, "tensorboard")
        if os.path.exists(log_dir):
            steps, values = extract_scalar_values(log_dir)
            if steps and values:
                plt.plot(steps, values, label=label)
    
    plt.xlabel("Steps")
    plt.ylabel("Cache Hit Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()

generate_plot(experiments_embedder, "LSTM Baseline vs Byte Embedder", "LSTM_Embedder_Comparison.png")
generate_plot(experiments_reuse, "LSTM Baseline vs Reuse Distance Loss Ablation", "LSTM_Reuse_Distance_Comparison.png")
generate_plot(experiments_ranking, "LSTM Baseline vs Ranking Loss Ablation", "LSTM_Ranking_Loss_Comparison.png")

print(f"Plots saved in {save_dir}")
