"""Module for reading, validating, and manipulating surface meshes."""

from trimesh import Trimesh
import numpy as np
from nibabel import load
from nibabel.freesurfer import read_geometry
from pathlib import Path
from typing import Union
from lapy import TriaMesh
from numpy.typing import NDArray, ArrayLike
from importlib.resources import files

def read_surf(
    mesh: Union[str, Path, Trimesh, TriaMesh, dict]
) -> Trimesh:
    """Load and validate a surface mesh.

    Parameters
    ----------
    mesh : str, Path, trimesh.Trimesh, lapy.TriaMesh, or dict
        Surface mesh specified as a file path (string or Path) to a VTK (.vtk), GIFTI (.gii), or
        FreeSurfer file (.white, .pial, .inflated, .orig, .sphere, .smoothwm, .qsphere, .fsaverage),
        an instance of trimesh.Trimesh or lapy.TriaMesh, or a dictionary with "vertices" and
        "faces" keys.

    Returns
    -------
    trimesh.Trimesh
        Validated surface mesh with vertices and faces.
    """
    if isinstance(mesh, Trimesh):
        trimesh = mesh
    elif isinstance(mesh, TriaMesh):
        trimesh = Trimesh(vertices=mesh.v, faces=mesh.t)
    elif isinstance(mesh, dict):
        trimesh = Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
    else:
        mesh_str = str(mesh)
        # check that file exists
        if not Path(mesh_str).is_file():
            raise ValueError(f'File not found: {mesh_str}')

        # Handle different file types
        if mesh_str.endswith('.vtk'):
            mesh_lapy = TriaMesh.read_vtk(mesh_str)
            trimesh = Trimesh(vertices=mesh_lapy.v, faces=mesh_lapy.t)
        elif mesh_str.endswith('.gii'):
            mesh_data = load(mesh_str).darrays
            trimesh = Trimesh(vertices=mesh_data[0].data, faces=mesh_data[1].data)
        elif mesh_str.endswith(
            ('white', 'pial', 'inflated', 'orig', 'sphere', 'smoothwm', 'qsphere', 'fsaverage')
            ):
            vertices, faces = read_geometry(mesh_str)
            trimesh = Trimesh(vertices=vertices, faces=faces)
        else:
            raise ValueError(
                '`surf` must be a path-like string to a valid VTK (.vtk), GIFTI (.gii), or '
                'FreeSurfer file (.white, .pial, .inflated, .orig, .sphere, .smoothwm, .qsphere, '
                '.fsaverage), an instance of trimesh.Trimesh or lapy.TriaMesh, or a dictionary of '
                '"faces" and "vertices".'
                )
    
    # Validate the mesh before returning
    check_surf(trimesh)

    return trimesh

def mask_surf(
    surf: Trimesh,
    mask: ArrayLike
) -> Trimesh:
    """Remove specified vertices and corresponding faces from the surface mesh. Returns a validated 
    trimesh.Trimesh object."""
    mask = np.asarray(mask, dtype=bool)

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

def fetch_surf(
    surf: str = 'midthickness',
    species: str = 'human',
    template: str = 'fsLR',
    density: str = '32k',
    hemi: str = 'L'
) -> dict[str, Union[Trimesh, NDArray]]:
    """Load cortical surface mesh and medial wall mask from nsbtools data directory into a dictionary."""
    data_dir = files('nsbtools.data')

    try:
        mesh = read_surf(
            data_dir / f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_{surf}.surf.gii'
            )
        medmask = load(
            data_dir / f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_medmask.label.gii'
            ).darrays[0].data.astype(bool)
        
        return {"mesh": mesh, "medmask": medmask}
    except Exception as e:
        raise ValueError(
            f"Surface data not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/nsbtools/tree/main/nsbtools/data/included_data.csv for a "
            "list of available surfaces."
            ) from e

def fetch_map(
    data: str,
    species: str = 'human',
    template: str = 'fsLR',
    density: str = '32k',
    hemi: str = 'L'
) -> NDArray:
    """Load data from nsbtools data directory."""
    data_dir = files('nsbtools.data')

    try:
        filename = f'sp-{species}_tpl-{template}_den-{density}_hemi-{hemi}_{data}.func.gii'
        return load(data_dir / filename).darrays[0].data
    
    except Exception as e:
        raise ValueError(
            f"Map '{filename}' not found. Please see {data_dir}/included_data.csv or "
            "https://github.com/NSBLab/nsbtools/tree/main/nsbtools/data/included_data.csv for a "
            "list of available data files."
        ) from e
        
