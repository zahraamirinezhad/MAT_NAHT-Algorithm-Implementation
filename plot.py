# import json
# import matplotlib.pyplot as plt

# # First experiment
# with open("naht_results/mpe-pp/open_train/poam-pqvmq_open/sacred/poam_baseline_seed=112361_07-31-11-47-40/1/metrics.json") as f:
#     metrics1 = json.load(f)

# steps1 = metrics1["test_return_mean"]["steps"]
# act_loss1 = metrics1["test_return_mean"]["values"]


# # Second experiment
# with open("naht_results/mpe-pp/open_train/poam-pqvmq_open/sacred/poam_baseline_seed=112361_07-31-12-31-17/1/metrics.json") as f:
#     metrics2 = json.load(f)

# steps2 = metrics2["test_return_mean"]["steps"]
# act_loss2 = metrics2["test_return_mean"]["values"]


# # Plot
# plt.plot(steps1, act_loss1, label="Experiment 1")
# plt.plot(steps2, act_loss2, label="Experiment 2")

# plt.xlabel("Environment Step")
# plt.ylabel("Action Loss")
# plt.title("ED Action Loss Comparison")

# plt.legend()
# plt.grid()

# plt.show()

import json
import matplotlib.pyplot as plt

# First experiment
# with open("naht_results/3sv5z/open_train/poam-pqvmq_open/sacred/poam_baseline_seed=112358_09-14-08-59-01-simple-poam-limited-2000000/1/metrics.json") as f:
#     metrics1 = json.load(f)

# steps1 = metrics1["advantage_mean"]["steps"]
# act_loss1 = metrics1["advantage_mean"]["values"]


# Second experiment
with open("naht_results/3sv5z/open_train/poam-pqvmq_open/sacred/poam_baseline_seed=112358_09-27-22-14-42-poam-compare/1/metrics.json") as f:
    metrics2 = json.load(f)

steps2 = metrics2["train_battle_won_mean"]["steps"]
act_loss2 = metrics2["train_battle_won_mean"]["values"]

# Second experiment
with open("naht_results/3sv5z/open_train/poam-pqvmq_open/sacred/poam_baseline_seed=112358_10-01-19-43-28/1/metrics.json") as f:
    metrics3 = json.load(f)

steps3 = metrics3["train_battle_won_mean"]["steps"]
act_loss3 = metrics3["train_battle_won_mean"]["values"]


# Plot
# plt.plot(steps1, act_loss1, label="Simple Poam")
plt.plot(steps2, act_loss2, label="Teammate Aware pre")
plt.plot(steps3, act_loss3, label="Teammate Aware new")

plt.xlabel("Environment Step")
plt.ylabel("test_return_mean")
# plt.title("ED Action Loss Comparison")

plt.legend()
plt.grid()

plt.show()

