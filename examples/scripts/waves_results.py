#%%
from pathlib import Path

import numpy as np
import pandas as pd

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


id = 1

study_name = f"waves-reproducibility_id-{id}"
DEMO_DIR = Path(__file__).parent.parent.resolve()

subj_ids = np.loadtxt(
    DEMO_DIR / "data" / "subject_ids.txt", dtype=str
)

rows = []
for subj_id in subj_ids[:31]:
    for visit in ["test", "retest"]:
        storage = JournalStorage(
            JournalFileBackend(
                str(DEMO_DIR / "results" / "waves" / f"id-{id}" / subj_id / visit / "optuna.journal.log")
            )
        )
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        rows.append({
            "subj_id": subj_id,
            "visit": visit,
            "alpha": study.best_params["alpha"],
            "r": study.best_params["r"],
            "edge_fc": study.best_trial.user_attrs["edge_fc_corr"],
            "node_fc": study.best_trial.user_attrs["node_fc_corr"]
        })

results_df = pd.DataFrame(rows)

#%%


import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(data=results_df, x="subj_id", y="alpha", hue="visit")

# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# param_mapping = {"alpha": r"$\alpha$", "r": r"$r_s$"}
# visit_mapping = {
#     "test": {
#         "label": "Test",
#         "color": "tab:blue"
#     },
#     "retest": {
#         "label": "Retest",
#         "color": "tab:red"
#     }
# }
# for i, param in enumerate(["alpha", "r"]):
#     # Keep only subjects present in every visit group (needed to draw paired lines)
#     # Filter the results dataframe to only include subjects with data in all visits
#     pivot = results_df.pivot(index="subj_id", columns="visit", values=param)
#     complete_subjs = pivot.dropna().index  # subjects with data in all visits
#     plot_df = results_df[results_df["subj_id"].isin(complete_subjs)].copy()

#     # enforce a consistent category order matching visits
#     plot_df["visit"] = pd.Categorical(plot_df["visit"], categories=["test", "retest"], ordered=True)
#     plot_df = plot_df.sort_values(["visit", "subj_id"]).reset_index(drop=True)
#     sns.swarmplot(
#         data=plot_df, x="visit", y=param, 
#         order=["test", "retest"],
#         ax=axs[i], 
#         size=2.5, 
#         zorder=2, 
#         facecolor="none",
#         edgecolor="tab:blue",
#         linewidth=1
#     )

    # fig.canvas.draw() # force the swarmplot to render so we can access the offsets
    # # extract swarm positions for each group, in subj_id order
    # group_positions = {}
    # for j, visit in enumerate(["test", "retest"]):
    #     # color each category's points differently
    #     axs[i].collections[j].set_edgecolor(visit_mapping[visit]["color"])

    #     # set the facecolor to none (transparent) so we can see the connecting lines
    #     sub = plot_df[plot_df["visit"] == visit]
    #     offsets = axs[i].collections[j].get_offsets()
    #     y_to_subj = dict(zip(np.round(sub[param].to_numpy(), 8), sub["subj_id"].to_numpy()))
    #     positions = {}
    #     for x, y in offsets:
    #         subj_id = y_to_subj[np.round(y, 8)]
    #         positions[subj_id] = (x, y)
    #     group_positions[visit] = positions

    # for subj_id in complete_subjs:
    #     xs, ys = [], []
    #     for visit in ["test", "retest"]:
    #         x, y = group_positions[visit][subj_id]
    #         xs.append(x)
    #         ys.append(y)
    #     axs[i].plot(xs, ys, color='gray', alpha=0.3, linewidth=1, zorder=1)

    # axs[i].set_ylabel("Pearson's r", fontsize=10)
    # axs[i].set_xlabel("", fontsize=10)
    # axs[i].set_xticklabels([visit_mapping[label]["label"] for label in ["test", "retest"]], fontsize=10)
    # axs[i].set_title(f"{param_mapping[param]}", fontsize=12)
    # axs[i].margins(x=0.1)
    # axs[i].spines['top'].set_visible(False)
    # axs[i].spines['right'].set_visible(False)

plt.savefig(str(DEMO_DIR / "scripts" / "test.png"), dpi=300, bbox_inches="tight")
