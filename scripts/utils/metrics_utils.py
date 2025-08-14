"""Debug for metric from saved D matrices"""

# Debugging and metric computation for saved distance matrices from PointFlow outputs
# Reference: https://github.com/stevenygd/PointFlow/issues/26

# from the point flow code output CD and paper, the reported metric is:
# - lgan_mmd-CD
# - lgan_cov-CD
# - 1-NN-CD-acc

import torch, numpy as np
import os, os.path as osp
from pprint import pprint
import pandas as pd


# Copy from PointFlow
def lgan_mmd_cov(all_dist: torch.Tensor) -> dict:
    """
    Computes LGAN-based MMD and coverage metrics from distance matrices.

    Parameters:
    all_dist (torch.Tensor): A tensor containing distance values between sampled points.

    Returns:
    dict: A dictionary containing lgan_mmd, lgan_cov, and lgan_mmd_smp values.
    """
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().reshape(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(
    Mxx: torch.Tensor, Mxy: torch.Tensor, Myy: torch.Tensor, k: int, sqrt: bool = False
) -> dict:
    """
    k-Nearest Neighbors metric computation for evaluation of point cloud generation.

    Parameters:
    Mxx, Mxy, Myy (torch.Tensor): Tensors representing pairwise distances within and between point clouds.
    k (int): The number of neighbors to consider for k-NN.
    sqrt (bool): Flag to determine whether to take the square root of distance values.

    Returns:
    dict: A dictionary containing precision, recall, accuracy, and other metrics.
    """
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat(
        (torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0
    )
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float("inf")
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False
    )

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        "tp": (pred * label).sum(),
        "fp": (pred * (1 - label)).sum(),
        "fn": ((1 - pred) * label).sum(),
        "tn": ((1 - pred) * (1 - label)).sum(),
    }

    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
            "acc": torch.eq(label, pred).float().mean(),
        }
    )
    return s


def eval_instantiation_distance(
    eval_directory: str,
    gen_name: str,
    ref_name: str,
    N_states: int = 10,
    N_pcl: int = 2048,
):
    """
    Evaluates the instantiation distance using saved distance matrices.

    Parameters:
    eval_dir (str): Directory where evaluation results are stored.
    gen_name (str): Name identifier for the generated point clouds.
    ref_name (str): Name identifier for the reference point clouds.
    N_states (int): Number of states in the evaluation.
    N_pcl (int): Number of points in each point cloud.

    Returns:
    None
    """
    # Directory to save the distance matrices
    results = {}
    ref = os.path.basename(ref_name)
    gen = os.path.basename(gen_name)
    rs_fn = os.path.join(
        eval_directory, f"ID_D_matrix/{gen}_{ref}_{N_states}_{N_pcl}.npz"
    )
    rr_fn = os.path.join(
        eval_directory, f"ID_D_matrix/{ref}_{ref}_{N_states}_{N_pcl}.npz"
    )
    ss_fn = os.path.join(
        eval_directory, f"ID_D_matrix/{gen}_{gen}_{N_states}_{N_pcl}.npz"
    )
    M_rs = torch.from_numpy(np.load(rs_fn)["D"])
    M_rr = torch.from_numpy(np.load(rr_fn)["D"])
    M_ss = torch.from_numpy(np.load(ss_fn)["D"])
    ret = lgan_mmd_cov(M_rs.t())
    results.update({"%s-ID" % k: v for k, v in ret.items()})
    ret = knn(M_rr, M_rs, M_ss, 1, sqrt=False)
    results.update({"1-NN-ID-%s" % k: v for k, v in ret.items() if "acc" in k})
    print(gen_name, ref_name)
    final_results = {
        "1-NN-ID-acc": results["1-NN-ID-acc"],
        "lgan_mmd-ID": results["lgan_mmd-ID"],
        "lgam_cov-ID": results["lgan_cov-ID"],
    }
    # final_results = {
    #     "1-NN-ID-acc": float(results["1-NN-ID-acc"]),
    #     "lgan_mmd-ID": float(results["lgan_mmd-ID"]),
    #     "lgam_cov-ID": float(results["lgan_cov-ID"]),
    # }
    pprint(final_results)

    # Create the LaTeX table string
    latex_table = create_latex_table(final_results)

    # Write the LaTeX table to a .tex file
    with open(os.path.join(eval_directory, gen + ".tex"), "w") as file:
        file.write(latex_table)

    print("LaTeX table has been saved to results_table.tex")

    return


def csvs_to_latex(
    base_path="/home/qleboute/Documents/git/midgard_internal/output/neurips",
    eval_dir="eval",
    metrics=["1-NN-ID-acc", "lgan_mmd-ID", "lgam_cov-ID"],
) -> None:
    """Iterates over a whole directory of results, concatenating them and printing the latex table"""
    all_res_csvs = []
    for mode in os.listdir(base_path):
        eval_path = os.path.join(base_path, mode, eval_dir, "results_table.tex")
        if not os.path.exists(eval_path):
            continue
        res = {}
        # load tex file
        with open(eval_path, "r") as inf:
            for l in inf.readlines():
                # get metrics from lines
                for m in metrics:
                    if m in l:
                        res[m] = (l.split("&")[-1]).replace("\\", "")
        # turn into a dataframe
        csv_res = pd.DataFrame(res, index=["tmp"]).reset_index().drop("index", axis=1)
        # # next lines: in case csvs are saved instead of .tex file
        # if not os.path.exists(os.path.join(eval_path, "result_table.csv")):
        #     continue
        # csv_res = pd.read_csv(os.path.join(eval_path, "result_table.csv"))
        csv_res["mode"] = mode  # e.g., neurips_sa1_sb1_j1_r1_m1_2d1_3d0_v2
        all_res_csvs.append(csv_res)
    all_res_df = pd.concat(all_res_csvs)
    # to reorder columns:
    # all_res_df = all_res_df[["lgan_mmd-ID", "lgam_cov-ID", "1-NN-ID-acc"]]
    print(all_res_df.set_index("mode").to_latex(float_format="%.2f"))


def create_csv(results, eval_directory):
    # save metrics to csv -> can later be converted to latex
    pd.DataFrame(results, index=["model"]).to_csv(
        os.path.join(eval_directory, "results_table.csv"), index=False
    )


def create_latex_table(data):
    # Function to create a LaTeX table from the nested dictionary
    table = "\\begin{table}[h!]\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{|c|c|c|}\n"
    table += "\\hline\n"
    table += "Category & Metric & Value \\\\\n"
    table += "\\hline\n"
    for category, metrics in data.items():
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    table += f"{category} & {key} & {value.item():.4f} \\\\\n"
                else:
                    table += f"{category} & {key} & {value} \\\\\n"
                table += "\\hline\n"
        elif isinstance(metrics, torch.Tensor):
            table += f"{category} & Value & {metrics.item():.4f} \\\\\n"
            table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Results Table}\n"
    table += "\\label{tab:results}\n"
    table += "\\end{table}\n"
    return table
