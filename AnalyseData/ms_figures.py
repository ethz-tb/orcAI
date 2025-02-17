#!/usr/bin/env python

# %%
import matplotlib.pyplot as plt
import json
import sys
import matplotlib.colors as mcolors
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import auxiliary as aux

# %%
# directories
project_dir = "/Users/sb/polybox/Documents/Research/Sebastian/OrcAI_project/"
os.chdir(project_dir)
data_dir = project_dir + "/Results/Euler/Final/"
msfigure_dir = project_dir + "AnalyseData/MSFigures/"

# %%
# load histories of model training


models = ["cnn_res_model", "cnn_res_lstm_model", "cnn_res_transformer_model"]

histories = {}
for mod in models:
    # Read the file content
    with open(data_dir + mod + "/" + "training_history.dict", "r") as file:
        raw_data = file.read()
    valid_json_data = raw_data.replace("'", '"')

    # Parse the modified string as JSON
    parsed_data = json.loads(valid_json_data)
    histories[mod] = parsed_data

# %%
# print max validation accuracy for each model
print("Maximal validation accuracy:")
for mod in models:
    print(
        "  - model:",
        mod,
        "accuracy:",
        np.around(100 * np.max(histories[mod]["val_masked_binary_accuracy"]), 1),
        "%",
    )

# %%
# Plot training and validation loss
plt.figure(figsize=(13, 6))

cols = list(mcolors.TABLEAU_COLORS.values())[0:3]  # solid cols
x_offset = 0
y_offset = 1.05
linestyles = {"loss": "dotted", "val_loss": "solid"}
models_colors = {
    "cnn_res_model": cols[0],
    "cnn_res_lstm_model": cols[1],
    "cnn_res_transformer_model": cols[2],
}
# Subplot 1: Training Loss
plt.subplot(1, 3, 1)
for i, mod in enumerate(histories.keys()):
    for type in ["loss", "val_loss"]:
        steps = range(1, len(histories[mod][type]) + 1)
        plt.plot(
            steps,
            histories[mod][type],
            label=mod,
            linestyle=linestyles[type],
            color=cols[i],
        )
plt.xlabel("Epochs")
plt.ylabel("Masked loss")
model_legend_handles = [
    plt.Line2D([0], [0], color=color, lw=2, label=model)
    for model, color in models_colors.items()
]
linestyles_1 = {"training": "dotted", "validation": "solid"}

style_legend_handles = [
    plt.Line2D(
        [0], [0], color="black", linestyle=linestyles_1[style], lw=2, label=style
    )
    for style in list(linestyles_1.keys())
]
first_legend = plt.legend(handles=model_legend_handles, loc="upper right")
plt.gca().add_artist(first_legend)  # Add the first legend to the plot
plt.legend(handles=style_legend_handles, loc="right")
plt.grid()
plt.text(
    x_offset,
    y_offset,
    "A",
    transform=plt.gca().transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)

# Subplot 2: Validation Accuracy
plt.subplot(1, 3, 2)
linestyles = {"masked_binary_accuracy": "dotted", "val_masked_binary_accuracy": "solid"}
for i, mod in enumerate(histories.keys()):
    for type in ["masked_binary_accuracy", "val_masked_binary_accuracy"]:
        steps = range(1, len(histories[mod][type]) + 1)
        plt.plot(
            steps,
            histories[mod][type],
            label=mod,
            linestyle=linestyles[type],
            color=cols[i],
        )
plt.xlabel("Epochs")
plt.ylabel("Masked accuracy")
# plt.title("Validation Accuracy")
# plt.legend()
plt.grid()
plt.text(
    x_offset,
    y_offset,
    "B",
    transform=plt.gca().transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)

# Subplot 3: learning rate
plt.subplot(1, 3, 3)
for i, mod in enumerate(histories.keys()):
    steps = range(1, len(histories[mod][type]) + 1)
    plt.plot(
        steps,
        histories[mod]["lr"],
        label=mod,
        linestyle="solid",
        color=cols[i],
    )
plt.xlabel("Epochs")
plt.ylabel("Learning rate")
# plt.legend()
plt.grid()
plt.text(
    x_offset,
    y_offset,
    "C",
    transform=plt.gca().transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
    ha="right",
)

plt.tight_layout()
plt.savefig(msfigure_dir + "train_history.pdf")
plt.show()

# %%
# Hyperparameter search
import json
import os
import pandas as pd

# %%
# Directory containing trials
trials_dir = data_dir + "cnn_res_lstm_model/hp_logs/lstm1/"

# Collect trial results
trial_data = []
for trial_id in os.listdir(trials_dir):
    trial_path = os.path.join(trials_dir, trial_id, "trial.json")
    if os.path.isfile(trial_path):
        with open(trial_path, "r") as f:
            data = json.load(f)
            trial_data.append(
                {**data["hyperparameters"]["values"], "score": data["score"]}
            )

# Convert to a DataFrame for analysis
df_trials = pd.DataFrame(trial_data)
df_trials.rename(columns={"score": "validation accuracy"}, inplace=True)

# Find the best trial
df_sorted = df_trials.sort_values(by="validation accuracy", ascending=False)
print("Best Hyperparameters:")
print(df_sorted.iloc[0])

# %%
# Create boxplots for each factor
import seaborn as sns
import matplotlib.pyplot as plt

# Factors to analyze
factors = ["filters", "lstm_units", "dropout_rate", "kernel_size"]

# Create subplots: 1 row, 4 columns
fig, axes = plt.subplots(1, len(factors), figsize=(10, 5), constrained_layout=True)

# Generate boxplots for each factor
for i, factor in enumerate(factors):
    sns.boxplot(x=factor, y="validation accuracy", data=df_trials, ax=axes[i])

# Show the plot
plt.savefig(project_dir + "AnalyseData/MSFigures/hyperparam_boxplot.pdf")
plt.show()


# %%
from sklearn.ensemble import RandomForestRegressor

# Fit Random Forest
rf = RandomForestRegressor()
df_trials["sum_filters"] = 0
df_trials.loc[df_trials["filters"] == "set1", "col"] = 100
df_trials.loc[df_trials["filters"] == "set2", "col"] = 140
rf.fit(
    df_trials[["sum_filters", "lstm_units", "dropout_rate", "kernel_size"]],
    df_trials["score"],
)

# Feature importance
feature_importances = rf.feature_importances_
importance_df = pd.DataFrame(
    {
        "Feature": ["sum_filters", "lstm_units", "dropout_rate", "kernel_size"],
        "Importance": feature_importances,
    }
)
importance_df = importance_df.sort_values("Importance", ascending=False)
print(importance_df)


# %%
