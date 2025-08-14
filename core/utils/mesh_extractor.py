import torch
import numpy as np
import platform
import mcubes
import open3d as o3d
from typing import Tuple, Optional


class MeshExtractor(object):
    """
    Mesh Generator based on Marching Cubes.
    """

    def __init__(self) -> None:
        self.implicit_F = None
        os_type = platform.system()
        if os_type == "Darwin":
            self.device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        elif os_type == "Linux" or os_type == "Windows":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.xpu.is_available():
                self.device = torch.device("xpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

    def _volume_to_mesh(
        self,
        volume: np.ndarray,
        level: float = 0.0,
        bbox_min: Optional[Tuple[float, float, float]] = None,
        bbox_max: Optional[Tuple[float, float, float]] = None,
    ):
        """
        Create a mesh from a 3D scalar field

        If a bounding box is provided along with the volume the mesh will be scaled
        and translated accordingly.

        Args:
            volume: A 3D array with memory order [z,y,x].
            level: The value at which to extract the contour
            bbox_min: Optional (3,) bounding box min values
            bbox_max: Optional (3,) bounding box max values
        Returns:
            The mesh as o3d.t.geometry.TriangleMesh
        """
        assert not ((bbox_min is None) ^ (bbox_max is None))
        surface = o3d.t.geometry.TriangleMesh.create_isosurfaces(volume, [level])
        if surface.vertex.positions.shape[0]:
            # shift by +0.5 because the voxel center is not a corner of the bounding box
            surface.vertex.positions += 0.5
            surface.triangle.indices = surface.triangle.indices[:, [2, 1, 0]]
            if bbox_min is not None:
                bbox_min = np.asarray(bbox_min, dtype=np.float32)
                bbox_max = np.asarray(bbox_max, dtype=np.float32)
                resolution = np.array(volume.shape[::-1])
                voxel_size = ((bbox_max - bbox_min) / resolution).astype(np.float32)
                surface.vertex.positions = (
                    surface.vertex.positions * voxel_size + bbox_min
                )
        return surface

    def generate_from_sdf(
        self,
        sdf,
        box_size: float = 1.0,
        level: float = 0.02,
        padding: float = 0.0,  # 0.1
        lib: str = "open3d",
    ):
        """
        Generate a mesh from a given SDF.

        Args:
            sdf (torch.Tensor): SDF tensor
            box_size (float): size of the bounding box
            level (float): level to extract the contour
            padding (float): padding to add to the bounding box
            lib (str): library to use for marching cubes

        Returns:
            Open3D TriangleMesh object
        """
        # Extract meshes from sdf
        n_cell = sdf.shape[-1]
        bs, nc = sdf.shape[:2]

        assert bs == 1, "The provided SDF has more than one channel!"
        assert nc == 1, "The provided SDF has more than one channel!"

        sdf_i = sdf[0, 0].detach().cpu().numpy()
        box_size = box_size + padding

        try:
            if lib == "mcubes":
                verts_i, faces_i = mcubes.marching_cubes(sdf_i, level)
                verts_i /= np.array([n_cell - 1, n_cell - 1, n_cell - 1])
                verts_i = box_size * (verts_i - 0.5)
                p3d_mesh = o3d.geometry.TriangleMesh()
                p3d_mesh.vertices = o3d.utility.Vector3dVector(verts_i)
                p3d_mesh.triangles = o3d.utility.Vector3iVector(faces_i)
                p3d_mesh.triangles = o3d.utility.Vector3iVector(
                    np.asarray(p3d_mesh.triangles)[:, ::-1]
                )
            elif lib == "open3d":
                bbox_min = 3 * (-0.5 * box_size,)
                bbox_max = 3 * (0.5 * box_size,)
                p3d_mesh = self._volume_to_mesh(
                    sdf_i,
                    level=level,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                ).to_legacy()
            else:
                raise ValueError("Unknown library for marching cubes!")

            # Compute the normals
            p3d_mesh.compute_vertex_normals()

            # Remove isolated triangle artifacts from the marching cube
            (
                triangle_clusters,
                cluster_n_triangles,
                cluster_area,
            ) = p3d_mesh.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < 50
            p3d_mesh.remove_triangles_by_mask(triangles_to_remove)

        except:
            p3d_mesh = None

        return p3d_mesh
