import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tensorboard_base_dir = r"C:\Users\Kahaan Patel\Desktop\Sem II\GenAI for Comp Sys\HW01\tensorboard"
save_dir = r"C:\Users\Kahaan Patel\Desktop\Sem II\GenAI for Comp Sys\HW01\graphs"
os.makedirs(save_dir, exist_ok=True)

belady_hit_rate = 0.3894  
lru_hit_rate = 0.044  

def extract_cache_hit_rate(log_dir):
    event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")]
    if not event_files:
        return None
    
    event_path = os.path.join(log_dir, event_files[0])
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()

    cache_hit_rate_tag = next((tag for tag in event_acc.Tags()["scalars"] if "cache_hit_rate" in tag), None)
    if not cache_hit_rate_tag:
        return None
    
    events = event_acc.Scalars(cache_hit_rate_tag)
    return events[-1].value if events else None

experiments = {
    "Section 3 Task 1 - Neurons": ["mlp_width_32", "mlp_width_64", "mlp_width_128", "mlp_width_256", "mlp_width_512"],
    "Section 3 Task 2 - Depth": ["mlp_depth_1", "mlp_depth_2", "mlp_depth_3", "mlp_depth_4"],
    "Section 3 Task 3 - Activation": ["mlp_ReLU", "mlp_leakyReLU", "mlp_sigmoid", "mlp_tanh"],
    "Section 3 Task 4 - Batch Size": ["mlp_batchsize_1", "mlp_batchsize_4", "mlp_batchsize_16", "mlp_batchsize_32", "mlp_batchsize_64"],
    "Section 3 Task 5 - Learning Rate": ["mlp_lr_0.00001", "mlp_lr_0.001", "mlp_lr_0.1"],
    "Section 4 Task 1 - RNN With Attention History": ["rnn_w_att_seq_20", "rnn_w_att_seq_40", "rnn_w_att_seq_60", "rnn_w_att_seq_80", "rnn_w_att_seq_100", "rnn_w_att_seq_120", "rnn_w_att_seq_140"],
    "Section 4 Task 2 - RNN Without Attention History": ["rnn_wo_att_seq_20", "rnn_wo_att_seq_40", "rnn_wo_att_seq_60", "rnn_wo_att_seq_80", "rnn_wo_att_seq_100", "rnn_wo_att_seq_120", "rnn_wo_att_seq_140"],
    "Section 3 Task 3 - RNN Max Attention History": ["rnn_w_att_hist_10", "rnn_w_att_hist_20", "rnn_w_att_hist_30", "rnn_w_att_hist_40", "rnn_w_att_hist_50"],
}

for task, exp_list in experiments.items():
    x_values, y_values = [], []
    is_numeric = True
    
    for exp in exp_list:
        log_dir = os.path.join(tensorboard_base_dir, exp, "tensorboard")
        if os.path.exists(log_dir):
            hit_rate = extract_cache_hit_rate(log_dir)
            if hit_rate is not None:
                param_value = exp.split("_")[-1]
                try:
                    param_value = int(param_value)
                except ValueError:
                    is_numeric = False
                x_values.append(param_value)
                y_values.append((hit_rate - lru_hit_rate) / (belady_hit_rate - lru_hit_rate))
    
    if x_values and y_values:
        plt.figure(figsize=(8, 4))
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
        
        for i, txt in enumerate(y_values):
            plt.annotate(f"{txt:.5f}", (x_values[i], y_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')
        
        plt.xlabel("Parameter")
        plt.ylabel("Normalized Cache Hit Rate")
        
        if is_numeric:
            x_values = sorted(x_values)
            plt.xticks(x_values, x_values)
        else:
            plt.xticks(range(len(x_values)), x_values)
        
        plt.title(f"{task} - Normalized Cache Hit Rate")
        plt.grid(True)
        
        save_path = os.path.join(save_dir, f"{task.replace(' ', '_')}.png")
        plt.savefig(save_path)
        plt.close()

print(f"Plots saved in {save_dir}")
