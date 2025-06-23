import numpy as np
import os
import torch
from tqdm import tqdm
from core.utils.common import chamfer_distance
# from pytorch3d.loss import chamfer_distance


@torch.no_grad()
def compute_instantiation_distance_pair(
    A: tuple,
    B: tuple,
    device: torch.device = torch.device("cuda:0"),
    N_states_max: int = 20,
    N_pcl_max: int = 2048,
    chunk: int = 1000,
) -> float:
    """
    Computes the instantiation distance between two sets of point clouds and poses.

    Parameters:
    A, B (tuple): Tuples containing point clouds and poses. Each tuple is expected to be in the format (pcl, pose).
    device (torch.device): The device (CPU/GPU) to perform calculations on.
    N_states_max (int): The maximum number of states to consider for each point cloud.
    N_pcl_max (int): The maximum number of points in each point cloud.
    chunk (int): The chunk size for processing in batches to manage memory usage.

    Returns:
    float: The computed instantiation distance.
    """
    # Unpack the point clouds and poses from the input tuples
    x1_list, pose1_list = A  # Shapes: (N,PCL,3), (N,P,4,4)
    x2_list, pose2_list = B  # Shapes: (N,PCL,3), (N,P,4,4)

    # Ensure the dimensions match and truncate to the specified max values
    P1, P2 = pose1_list.shape[1], pose2_list.shape[1]
    N_states, N_pcl = x1_list.shape[0], x1_list.shape[1]
    assert N_states == x2_list.shape[0] and N_pcl == x2_list.shape[1]
    N_states = min(N_states, N_states_max)
    N_pcl = min(N_pcl, N_pcl_max)
    x1_list = x1_list[:N_states, :N_pcl, :]
    x2_list = x2_list[:N_states, :N_pcl, :]
    pose1_list = pose1_list[:N_states]
    pose2_list = pose2_list[:N_states]

    # Move data to the specified device (GPU/CPU)
    x1_list, pose1_list = x1_list.float().to(device), pose1_list.float().to(device)
    x2_list, pose2_list = x2_list.float().to(device), pose2_list.float().to(device)

    # Canonicalize the point clouds to all possible poses
    x1_list = torch.cat(
        [x1_list, torch.ones_like(x1_list[..., :1])], dim=-1
    )  # N_states, PCL,4
    x2_list = torch.cat([x2_list, torch.ones_like(x2_list[..., :1])], dim=-1)
    inv_pose1_list = torch.inverse(pose1_list)  # N_states, P,4,4
    inv_pose2_list = torch.inverse(pose2_list)

    # Apply transformations
    x1_list = torch.einsum("npij,ntj->npti", inv_pose1_list, x1_list)[..., :3]
    x2_list = torch.einsum("npij,ntj->npti", inv_pose2_list, x2_list)[..., :3]

    # Compute the distance matrix between all pairs of states
    D = []
    cur = 0
    while cur < N_states:
        # Prepare all computing pairs
        src = x1_list[cur : cur + chunk]  # Shape: (chunk, P1, PCL, 3)
        dst = x2_list.to(device)  # Shape: (N_states, P2, PCL, 3)
        src = src[:, None, :, None, ...].expand(-1, N_states, -1, P2, -1, -1)
        dst = dst[None, :, None, ...].expand(len(src), -1, P1, -1, -1, -1)

        # Calculate Chamfer distance for each pair
        cd = chamfer_distance(src.reshape(-1, N_pcl, 3), dst.reshape(-1, N_pcl, 3))
        # cd, _ = chamfer_distance(src.reshape(-1, N_pcl, 3), dst.reshape(-1, N_pcl, 3), batch_reduction=None)
        cd = cd.reshape(len(src), N_states, -1).min(dim=-1).values

        # Try all canonicalization and find the best one
        D.append(cd)
        cur += chunk
    D = torch.cat(D, dim=0)  # Final distance matrix shape: (N_states, N_states)

    # Calculate the mean distance for each row and column, then sum them
    dl, dr = D.min(dim=1).values, D.min(dim=0).values
    dl, dr = dl.mean(), dr.mean()
    distance = dl + dr

    return float(distance.cpu().numpy())


def compute_D_matrix(
    gen_dir: str,
    ref_dir: str,
    save_dir: str,
    N_states_max: int = 10,
    N_pcl_max: int = 2048,
) -> np.ndarray:
    """
    Computes and saves a matrix of instantiation distances between point clouds from two directories.

    Parameters:
    gen_dir (str): Directory containing generated point clouds (.npz files).
    ref_dir (str): Directory containing reference point clouds (.npz files).
    save_dir (str): Directory to save the resulting distance matrix.
    N_states_max (int): Maximum number of states to consider for each point cloud.
    N_pcl_max (int): Maximum number of points in each point cloud.

    Returns:
    numpy.ndarray: The computed distance matrix.
    """

    # Modify the source directory path to create the destination directory
    path_parts = gen_dir.split(os.sep)
    path_parts[path_parts.index("G")] = "PCL"
    gen_dir = os.sep.join(path_parts)

    path_parts = ref_dir.split(os.sep)
    path_parts[path_parts.index("G")] = "PCL"
    ref_dir = os.sep.join(path_parts)

    # Retrieve and sort filenames from the directories
    gen_fn_list = [f for f in os.listdir(gen_dir) if f.endswith(".npz")]
    ref_fn_list = [f for f in os.listdir(ref_dir) if f.endswith(".npz")]
    gen_fn_list.sort()
    ref_fn_list.sort()

    # Initialize the distance matrix
    N_gen, N_ref = len(gen_fn_list), len(ref_fn_list)
    D = -1.0 * np.ones((N_gen, N_ref), dtype=np.float32)
    gen_name = os.path.basename(gen_dir)
    ref_name = os.path.basename(ref_dir)
    save_name = f"{gen_name}_{ref_name}_{N_states_max}_{N_pcl_max}.npz"
    save_fn = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"save to {save_fn}")

    # Cache point cloud data from both directories
    sym_flag = gen_dir == ref_dir
    DATA_GEN, DATA_REF = [], []

    # Caching generated data
    print("caching GEN ...")
    for i in tqdm(range(N_gen)):
        fn = os.path.join(gen_dir, gen_fn_list[i])
        data = np.load(fn)
        pcl, pose = torch.from_numpy(data["pcl"]), torch.from_numpy(data["pose"])
        DATA_GEN.append((pcl, pose))

    # Caching reference data (can be same as generated if sym_flag is True)
    if sym_flag:
        DATA_REF = DATA_GEN
    else:
        print("caching REF ...")
        for i in tqdm(range(N_ref)):
            fn = os.path.join(ref_dir, ref_fn_list[i])
            data = np.load(fn)
            pcl, pose = torch.from_numpy(data["pcl"]), torch.from_numpy(data["pose"])
            DATA_REF.append((pcl, pose))

    # Compute distances for all pairs
    for i in tqdm(range(N_gen)):
        for j in tqdm(range(N_ref)):
            if sym_flag and i == j:
                D[i, j] = 0.0
                continue
            if sym_flag and i > j:
                assert D[j, i] >= 0.0
                D[i, j] = D[j, i]
                continue
            pcl1, pose1 = DATA_GEN[i]
            pcl2, pose2 = DATA_REF[j]
            _d = compute_instantiation_distance_pair(
                (pcl1, pose1),
                (pcl2, pose2),
                N_states_max=N_states_max,
                N_pcl_max=N_pcl_max,
                chunk=1000,
            )
            D[i, j] = _d

    # Save the computed distance matrix
    assert (D >= 0.0).all(), "invalid D"
    np.savez_compressed(
        save_fn,
        D=D,
        gen_dir=gen_dir,
        ref_dir=ref_dir,
        N_states_max=N_states_max,
        N_pcl_max=N_pcl_max,
    )
    np.savez_compressed(
        save_fn,
        D=D,
        gen_dir=gen_dir,
        ref_dir=ref_dir,
        N_states_max=N_states_max,
        N_pcl_max=N_pcl_max,
        gen_fn_list=gen_fn_list,
        ref_fn_list=ref_fn_list,
    )

    return D
