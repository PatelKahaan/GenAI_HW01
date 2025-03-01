import os
import matplotlib.pyplot as plt
import numpy as np
import math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

main_dir = r"C:\Users\Kahaan Patel\Desktop\Sem II\GenAI for Comp Sys\HW01\tensorboard"
save_dir = r"C:\Users\Kahaan Patel\Desktop\Sem II\GenAI for Comp Sys\HW01\graphs"

os.makedirs(save_dir, exist_ok=True)

def tensorboard_smoothing(scalars, weight=0.6):
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        debias_weight = 1 if weight == 1 else (1 - math.pow(weight, num_acc))
        smoothed.append(last / debias_weight)
    return smoothed

tensorboard_dirs = []
for root, dirs, _ in os.walk(main_dir):
    if "tensorboard" in dirs:
        tensorboard_dirs.append(os.path.join(root, "tensorboard"))

for log_dir in tensorboard_dirs:
    event_files = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")],
        key=lambda x: os.path.getmtime(os.path.join(log_dir, x)),
        reverse=True
    )

    if not event_files:
        print(f"No TensorBoard logs found in {log_dir}")
        continue

    event_path = os.path.join(log_dir, event_files[0])
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()

    cache_hit_rate_tag = next((tag for tag in event_acc.Tags()["scalars"] if "cache_hit_rate" in tag), None)
    if not cache_hit_rate_tag:
        print(f"No 'cache_hit_rate' metric found in {log_dir}")
        continue

    events = event_acc.Scalars(cache_hit_rate_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    smoothed_values = tensorboard_smoothing(values, weight=0.6)

    plt.figure(figsize=(10, 5))
    experiment_name = os.path.basename(os.path.dirname(log_dir))

    plt.plot(steps, values, label="cache_hit_rate (Value)", alpha=0.5, color="blue")
    plt.plot(steps, smoothed_values, label="cache_hit_rate (Smoothed)", color="orange")

    textstr = f"Smoothed: {smoothed_values[-1]:.5f}\nValue: {values[-1]:.4f}\nStep: {steps[-1]:,}"
    plt.text(0.72, 0.15, textstr, transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.title(f"Cache Hit Rate - {experiment_name}")
    plt.xlabel("Steps")
    plt.ylabel("Cache Hit Rate")
    plt.legend(loc="lower right")
    plt.grid(True)

    save_path = os.path.join(save_dir, f"{experiment_name}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved graph: {save_path}")