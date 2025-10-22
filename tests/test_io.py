import numpy as np
from trimesh import Trimesh
from pytest import raises
from nsbtools.io import check_surf, load_data

def test_surf_unreferenced_verts():
    # Create an invalid mesh with unreferenced vertices
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2]])  # Last vertex unreferenced
    faces = np.array([[0, 1, 2], [0, 2, 3]])  # Only uses first 4 vertices, vertex 4 is unreferenced
    
    # IMPORTANT: Use process=False to prevent trimesh from automatically cleaning up unreferenced vertices    
    invalid_mesh = Trimesh(vertices=vertices, faces=faces, process=False)
   
    # check_surf should raise ValueError due to unreferenced vertex
    with raises(ValueError, match="Surface mesh contains .* unreferenced vertices"):
        check_surf(invalid_mesh)

def test_surf_not_contiguous():
    # Create two separate triangles (disconnected components)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])  # Two separate triangles
    disconnected_mesh = Trimesh(vertices=vertices, faces=faces)
    
    # check_surf should raise ValueError due to multiple components
    with raises(ValueError, match="Surface mesh is not contiguous.*connected components"):
        check_surf(disconnected_mesh)

def test_load_surf():
    mesh = load_data('surf')
    assert isinstance(mesh, Trimesh)
    assert mesh.vertices.shape == (32492, 3)

def test_load_medmask():
    medmask = load_data('medmask', species='marmoset', density='10k')
    assert isinstance(medmask, np.ndarray)
    assert medmask.dtype == bool
    assert medmask.shape == (10242,)

def test_load_gradient():
    grad = load_data('fcgradient1')
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (32492,)

def test_load_invalid_type():
    with raises(ValueError, match="Data file 'sp-human_tpl-fsLR_den-32k_hemi-L_panshifu.func.gii'.*"):
        load_data('panshifu')
