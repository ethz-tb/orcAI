import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_spec_and_labels(spec, calls_list, lab_true=None, lab_pred=None, title=None):

    if lab_pred is None:
        lab_pred = np.zeros(lab_true.shape).astype(int)
    if lab_true is None:
        lab_pred = np.zeros(lab_pred.shape).astype(int)
    if lab_pred.shape != lab_true.shape:
        print("WARNING: shapes of lab_pred and lab_true not equal")
        return
    # generate interleaved array with predicted and true labels
    lab_all = np.zeros((lab_pred.shape[0], 2 * lab_pred.shape[1])).astype(int)
    lab_all[:, 0::2] = lab_true  # Place rows of lab_true at even indices
    lab_all[:, 1::2] = lab_pred  # Place rows of array2 at odd indices

    cols_masked_absent = ["grey", "white"]
    cols_masked_absent = [(0.1, 0.1, 0.1, 0.1), (1, 1, 1, 1)]
    if lab_true.shape[1] < 10:
        cols_labels = list(mcolors.TABLEAU_COLORS.values())[
            0 : lab_true.shape[1]
        ]  # solid cols
    else:
        print("WARNING: More labels then colors")
    cols_labels_true_pred = []
    for color in cols_labels:
        cols_labels_true_pred += [
            mcolors.to_rgba(color, alpha=1),
            mcolors.to_rgba(color, alpha=0.5),
        ]  # generate pairs of solid and transparent colors
    cols = cols_masked_absent + cols_labels_true_pred
    cmap_lab = ListedColormap(cols)
    norm_lab = BoundaryNorm(
        np.linspace(-1.5, lab_all.shape[1] + 0.5, lab_all.shape[1] + 3), len(cols)
    )
    lab_arr = lab_all * np.linspace(1, lab_all.shape[1], lab_all.shape[1]).astype(int)

    plt.clf()
    fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 6))
    # fig, axs = plt.subplots(2, 1)
    axs[0].matshow(spec.T, origin="lower", cmap="gray_r")
    axs[0].set_aspect(aspect="auto")

    axs[1].matshow(lab_arr.T, origin="lower", cmap=cmap_lab, norm=norm_lab)
    axs[1].set_aspect(aspect="auto")

    yticks = (
        np.linspace(1, lab_arr.shape[1] - 1, lab_arr.shape[1] // 2) - 0.5
    )  # Tick positions
    for yt in yticks:
        axs[1].axhline(
            y=yt, color="black", linestyle="dotted", linewidth=0.5
        )  # Horizontal line black
        axs[1].axhline(
            y=yt - 0.95, color="white", linestyle="solid", linewidth=2
        )  # Horizontal line wite

    ytick_labels = calls_list  # Custom tick labels
    _ = axs[1].set_yticks(yticks)
    _ = axs[1].set_yticklabels(ytick_labels)
    _ = axs[1].set_xticks([])
    _ = axs[0].set_xticks([])
    _ = axs[0].set_yticks([])
    if not title is None:
        axs[0].set_title(title)
    axs[1].set_title("solid: annotation, transparent: prediction")
    plt.subplots_adjust(hspace=0.2)
    # plt.tight_layout()
    return
