import numpy as np
from trimesh import Trimesh
from pytest import raises
from nsbtools.io import check_surf, load_data, read_surf
from importlib.resources import files

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
    medmask = load_data('medmask', species='marmoset', density='38k')
    assert isinstance(medmask, np.ndarray)
    assert medmask.dtype == bool
    assert medmask.shape == (37974,)

def test_load_gradient():
    grad = load_data('fcgradient1')
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (32492,)

def test_load_invalid_type():
    with raises(ValueError, match="Data file 'sp-human_tpl-fsLR_den-32k_hemi-L_panshifu.func.gii'.*"):
        load_data('panshifu')

def test_read_surf_dict():
    mesh_data = {
        'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        'faces': [[0, 1, 2], [1, 2, 3]]
    }
    mesh = read_surf(mesh_data)
    assert isinstance(mesh, Trimesh)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (2, 3)
    
    mesh_data_numpy = {
        'vertices': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        'faces': np.array([[0, 1, 2], [1, 2, 3]])
    }
    mesh = read_surf(mesh_data_numpy)
    assert isinstance(mesh, Trimesh)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (2, 3)

    invalid_data = {
        'faces': np.array([[0, 1, 2]])
    }
    with raises(KeyError, match="'vertices'"):
        read_surf(invalid_data)

    invalid_data = {
        'vertices': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    }
    with raises(KeyError, match="'faces'"):
        read_surf(invalid_data)

def test_read_surf_vtk():
    vtk_path = files('nsbtools.data') / 'fsaverage5_10k_midthickness-lh.vtk'
    vtk_mesh = read_surf(vtk_path)

    assert isinstance(vtk_mesh, Trimesh)
    assert vtk_mesh.vertices.shape == (10242, 3)
    assert vtk_mesh.faces.shape == (20480, 3)

def test_read_surf_invalid():
    invalid_path = files('nsbtools.data') / 'invalid.surf.vtk'
    with raises(ValueError, match="not a valid file path"):
        read_surf(invalid_path)
