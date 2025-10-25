"""Module for reading, validating, and manipulating surface meshes."""

from trimesh import Trimesh
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union
from lapy import TriaMesh
from numpy.typing import NDArray, ArrayLike
from importlib.resources import files

def read_surf(
    surf: Union[str, Path, Trimesh, TriaMesh, dict]
) -> Trimesh:
    """Validate surface type and load if a file name. Returns a validated trimesh.Trimesh object."""
    if isinstance(surf, Trimesh):
        mesh = surf
    elif isinstance(surf, TriaMesh):
        mesh = Trimesh(vertices=surf.v, faces=surf.t)
    elif isinstance(surf, dict):
        mesh = Trimesh(vertices=surf['vertices'], faces=surf['faces'])
    else:
        surf_str = str(surf)
        # check that file exists
        if not Path(surf_str).is_file():
            raise ValueError('Converted string is not a valid file path.')
        if surf_str.endswith('.vtk'):
            mesh_data = TriaMesh.read_vtk(surf_str)
            mesh = Trimesh(vertices=mesh_data.v, faces=mesh_data.t)
        else:
            try:
                mesh_data = nib.load(surf_str).darrays
                mesh = Trimesh(vertices=mesh_data[0].data, faces=mesh_data[1].data)
            except Exception as e:
                raise ValueError(
                    '`surf` must be a path-like string to a valid nibabel-supported file (e.g., VTK'
                    ', GIFTI), or an instance of either trimesh.Trimesh or lapy.TriaMesh.'
                    ) from e
    
    # Validate the mesh before returning
    check_surf(mesh)

    return mesh

def mask_surf(
    surf: Trimesh,
    mask: ArrayLike
) -> tuple[Trimesh, NDArray]:
    """Remove specified vertices and corresponding faces from the surface mesh. Returns a validated 
    trimesh.Trimesh object."""
    if len(mask) != surf.vertices.shape[0]:
        raise ValueError(f"The number of elements in `mask` ({len(mask)}) must match "
                         f"the number of vertices in the surface mesh ({surf.vertices.shape[0]}).")
    
    # Mask faces where all vertices are in the mask
    face_mask = np.all(mask[surf.faces], axis=1)
    mesh = surf.submesh([face_mask])[0]

    # Validate the mesh before returning
    check_surf(mesh)
    
    return mesh

def check_surf(
    surf: Trimesh
) -> None:
    """Check if the surface mesh is contiguous with no unreferenced vertices."""

    # Check for unreferenced vertices
    referenced = np.zeros(len(surf.vertices), dtype=bool)
    referenced[surf.faces] = True
    if not np.all(referenced):
        raise ValueError(f'Surface mesh contains {np.sum(~referenced)} unreferenced '
                         'vertices (i.e., not part of any face).')

    # Check if the mesh is contiguous
    n_components = surf.body_count
    if n_components != 1:
        raise ValueError(f'Surface mesh is not contiguous: {n_components} connected components '
                         'found.')

def load_data(
    data_type: str,
    species: str = 'human',
    template: str = 'fsLR',
    density: str = '32k',
    hemi: str = 'L'
) -> Union[NDArray, Trimesh]:
    """Load data from nsbtools data directory."""
    data_dir = files('nsbtools.data')

    if data_type == 'surf':
        surf_path = data_dir / f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_midthickness.surf.gii'
        return read_surf(surf_path)
    elif data_type == 'medmask':
        medmask_path = data_dir / f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_medmask.label.gii'
        return nib.load(medmask_path).darrays[0].data.astype(bool)
    else:
        try:
            filename = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_{data_type}.func.gii'
            return nib.load(data_dir / filename).darrays[0].data
        except Exception as e:
            raise ValueError(f"Data file '{filename}' not found. Please see {data_dir}/included_dat"
                             "a.csv or https://github.com/NSBLab/nsbtools/tree/main/nsbtools/data/i"
                             "ncluded_data.csv for a list of available data files.") from e