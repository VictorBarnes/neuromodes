import io
import numpy as np
from scipy.ndimage import label
from numpy.linalg import norm
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyvista as pv
from PIL import Image

camera_views = {
    'lateral':   dict(position=(-1, 0, 0), viewup=(0, 0, 1)),
    'medial':    dict(position=( 1, 0, 0), viewup=(0, 0, 1)),
    'dorsal':    dict(position=( 0, 0, 1), viewup=(0, 1, 0)),
    'ventral':   dict(position=( 0, 0,-1), viewup=(0, 1, 0)),
    'anterior':  dict(position=( 0, 1, 0), viewup=(0, 0, 1)),
    'posterior': dict(position=( 0,-1, 0), viewup=(0, 0, 1)),
}

def plot_brain_pyvista(mesh, data=None, size=(800, 600),
                       color='lightgrey', cmap='viridis',
                       smooth_shading=True,
                       background='white',
                       view='lateral',
                       zoom=1.0,
                       off_screen=False,
                       clim=None):
    """
    Plot a single brain mesh in PyVista with optional per-vertex data overlay
    and selectable camera view.

    Parameters
    ----------
    mesh : dict
        Dictionary with keys 'vertices' (Nx3 array) and 'faces' (Mx3 array).
    data : array_like, optional
        Per-vertex data for coloring the mesh.
    view : str
        One of the keys in `camera_views`.
    off_screen : bool
        Whether to render off-screen (required for video creation).
    clim : tuple, optional
        Color limits (vmin, vmax) for consistent scaling across frames.
    Other parameters control lighting, shading, opacity, etc.
    
    Returns
    -------
    plotter : pv.Plotter
        The PyVista plotter instance.
    pv_mesh : pv.PolyData
        The PyVista mesh object (for updating data).
    """
    verts = mesh.vertices
    verts_centered = verts - np.mean(verts, axis=0)
    faces = mesh.faces

    # Convert faces to PyVista format
    faces_vtk = np.hstack([np.full((faces.shape[0],1),3), faces]).astype(np.int64)
    pv_mesh = pv.PolyData(verts_centered, faces_vtk)

    if data is not None:
        if len(data) != pv_mesh.n_points:
            raise ValueError("Data length must match number of vertices")
        pv_mesh.point_data['data'] = data

    plotter = pv.Plotter(window_size=size, off_screen=off_screen)
    plotter.background_color = background
    plotter.parallel_projection = True

    # Add mesh
    mesh_kwargs = dict(
        smooth_shading=smooth_shading,
        show_scalar_bar=True if data is not None else False
    )
    
    if data is not None:
        mesh_kwargs.update(dict(
            scalars='data',
            cmap=cmap,
            clim=clim if clim is not None else [np.min(data), np.max(data)]
        ))
    else:
        mesh_kwargs.update(dict(color=color))
        
    actor = plotter.add_mesh(pv_mesh, **mesh_kwargs)

    if view not in camera_views:
        raise ValueError(f"View '{view}' not recognised")

    max_range = np.array([verts_centered[:,0].max() - verts_centered[:,0].min(),
                          verts_centered[:,1].max() - verts_centered[:,1].min(),
                          verts_centered[:,2].max() - verts_centered[:,2].min()]).max()
    print(max_range)
    pos = np.array(camera_views[view]['position']) * max_range
    viewup = camera_views[view]['viewup']
    cam = plotter.camera
    cam.SetPosition(pos)
    cam.SetFocalPoint((0, 0, 0))
    cam.SetViewUp(viewup)
    if zoom is not None:
        cam.zoom(zoom)

    # Adjust lighting properties
    prop = actor.GetProperty()
    prop.SetAmbient(0.1)            # 0.0 – 1.0
    prop.SetDiffuse(0.6)            # 0.0 – 1.0
    prop.SetSpecular(0)          # 0.0 – 1.0
    prop.SetSpecularPower(10)

    # Lock light to camera
    plotter.renderer.lights.clear()
    cam_light = pv.Light()
    cam_light.set_camera_light()
    plotter.renderer.add_light(cam_light)

    return plotter, pv_mesh


def create_brain_video(mesh, data_timeseries, filename='brain_animation.mp4',
                      framerate=10, view='lateral', cmap='plasma',
                      size=(800, 608), title_template="Time: {:.1f} ms"):
    """
    Create a video of brain activity over time.
    
    Parameters
    ----------
    mesh : dict
        Dictionary with keys 'vertices' and 'faces'.
    data_timeseries : array_like
        2D array of shape (n_vertices, n_timepoints).
    filename : str
        Output video filename.
    framerate : int
        Video framerate.
    view : str
        Camera view.
    cmap : str
        Colormap.
    size : tuple
        Video dimensions.
    title_template : str
        Template for frame titles with {} for time formatting.
        
    Returns
    -------
    str
        Path to created video file.
    """
    data_timeseries = np.asarray(data_timeseries)
    n_verts, n_frames = data_timeseries.shape
    
    # Get global color limits for consistency
    abs_max = np.max(np.abs(data_timeseries))
    clim = [-abs_max, abs_max]
    
    # Create plotter and mesh
    plotter, pv_mesh = plot_brain_pyvista(
        mesh, 
        data=data_timeseries[:, 0],
        view=view,
        cmap=cmap,
        size=size,
        off_screen=True,
        clim=clim
    )
    
    # Setup video recording
    plotter.open_movie(filename, framerate=24)
    plotter.show(auto_close=False)  # only necessary for an off-screen movie
    plotter.write_frame()  # write initial data

    # Animate through frames
    for frame in range(n_frames):
        # Update mesh data
        pv_mesh.point_data['data'] = data_timeseries[:, frame]
        
        # Add title
        if title_template:
            plotter.add_text(
                title_template.format(frame * (1000/framerate)),  # Assume ms timing
                position='upper_left',
                font_size=16,
                name='time_text'
            )
        
        # Write frame
        plotter.write_frame()
        
        # Clear text for next frame
        if title_template:
            plotter.remove_actor('time_text')
    
    # Finalize video
    plotter.close()
    
    return filename



def plot_brain_plotly(
        mesh,
        data=None,
        rois=None,
        views=['lateral', 'medial'],
        layout='grid',
        size=(800, 600),
        zoom=2.0,
        cbar=False,
        cmap='turbo',
        mesh_edges=False, 
        roi_outlines=False,
    ) -> go.Figure:
    """
    Plot brain surface with optional data overlay using Plotly.
    
    Parameters:
    -----------
    surfs : dictionary-like
        ...
    data : array_like, optional
        ...
    rois : dict[str, np.ndarray], optional
        Parcellation of surface; dictionary with keys 'lh' and/or 'rh'.
        Each value is a vector of length n_vertices containing integer ROI IDs.
        Vertices labeled 0 are considered unallocated (e.g., medial wall)
        and will be shown in the base surface color (lightgray).
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    
    hemis = list(mesh.keys())
    n_hemi = len(hemis)
    n_views = len(views)

    # Determine subplot grid size
    if layout == 'grid':
        rows, cols = n_views, n_hemi
    elif layout == 'row':
        rows, cols = 1, n_hemi * n_views
    elif layout == 'column':
        rows, cols = n_hemi * n_views, 1
    else:
        raise ValueError("`layout` must be one of 'row', 'column', or 'grid'.")

    fig = make_subplots(    
        rows=rows, cols=cols,
        specs=[[{'type': 'scene'}]*cols]*rows,
        horizontal_spacing=0.01,
        vertical_spacing=0.01
    )

    # Loop through hemispheres and views
    for h_idx, hemi in enumerate(hemis):
        surf = mesh[hemi]
        verts = np.array(surf['v'])
        n_verts = verts.shape[0]
        tris = np.array(surf['t'])
        n_tris = tris.shape[0]
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = tris[:, 0], tris[:, 1], tris[:, 2]

        # Right hemisphere needs to flip medial/lateral views
        if hemi == 'rh':
            view_key = dict(medial='lateral', lateral='medial', dorsal='dorsal', 
                        ventral='ventral', anterior='anterior', 
                        posterior='posterior')
            views = [view_key[v] for v in views]

        # Data for this hemisphere
        hemi_data = None
        if data is not None:
            if isinstance(data, dict):
                hemi_data = data.get(hemi, None)
            else:
                hemi_data = data

        if rois is not None:
            roi_labels = rois[hemi]
            mask = np.where(roi_labels == 0, False, True)

            if hemi_data is not None:
                n_rois = int(np.max(roi_labels)) # int(np.unique_counts(roi_labels))
                if len(hemi_data) == n_rois:
                    # Map ROI data to vertices
                    vertex_data = np.zeros(len(roi_labels))
                    for roi_id in range(1, n_rois + 1):
                        vertex_data[roi_labels == roi_id] = hemi_data[roi_id - 1]
                    hemi_data = vertex_data
        else:
            mask = np.ones(n_verts, dtype=bool)

        for v_idx, view in enumerate(views):
            # Determine subplot row/col
            if layout in ['row', 'column']:
                r, c = 1, h_idx * n_views + v_idx + 1 if layout=='row' else h_idx * n_views + v_idx + 1
            else:  # grid
                r, c = v_idx+1, h_idx+1
        
            # Create mesh
            mesh_kwargs = dict(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                flatshading=False,
                lighting=dict(
                    ambient=0.2,  # lower ambient = less overall light
                    diffuse=1,     # full diffuse = full light reflection
                    specular=0.1,  # very low specular = almost no shine
                    roughness=0,   # no roughness
                    fresnel=0      # no fresnel effect
                ),
                lightposition=dict(x=0, y=0, z=-1e5),
                intensitymode='vertex',
            )
            if hemi_data is not None:
                # If masked, create 2 meshes: one for masked, one for unmasked
                if mask is not None:
                    tris_data = tris[np.all(mask[tris], axis=1)]
                    tris_mask = tris[~np.all(mask[tris], axis=1)]

                mesh_data_kwargs = mesh_kwargs.copy()
                mesh_data_kwargs['i'] = tris_data[:, 0]
                mesh_data_kwargs['j'] = tris_data[:, 1]
                mesh_data_kwargs['k'] = tris_data[:, 2]
                mesh_data_kwargs['intensity'] = hemi_data
                mesh_data_kwargs['colorscale'] = cmap
                mesh_data_kwargs['showscale'] = cbar
                # TODO: work out how to set colorbar for entire figure or just one surf
                if cbar:
                    mesh_data_kwargs['colorbar'] = dict(len=0.5)
                mesh_data_kwargs['name'] = f'data-{hemi}-{v_idx}'
                fig.add_trace(go.Mesh3d(**mesh_data_kwargs), row=r, col=c)

                mesh_mask_kwargs = mesh_kwargs.copy()
                mesh_mask_kwargs['i'] = tris_mask[:, 0]
                mesh_mask_kwargs['j'] = tris_mask[:, 1]
                mesh_mask_kwargs['k'] = tris_mask[:, 2]
                mesh_mask_kwargs['color'] = "lightgrey"
                mesh_mask_kwargs['showscale'] = False
                mesh_mask_kwargs['name'] = f'mask-{hemi}-{v_idx}'
                fig.add_trace(go.Mesh3d(**mesh_mask_kwargs), row=r, col=c)
                
            else:
                mesh_kwargs['color'] = "lightgrey"
                mesh_kwargs['name'] = f'mesh-{hemi}-{v_idx}'
                fig.add_trace(go.Mesh3d(**mesh_kwargs), row=r, col=c)

            # Add edges
            if mesh_edges:
                xe = verts[tris[:, [0,1,2,0]], 0].flatten()
                ye = verts[tris[:, [0,1,2,0]], 1].flatten()
                ze = verts[tris[:, [0,1,2,0]], 2].flatten()
                xe_sep, ye_sep, ze_sep = [], [], []
                for idx in range(0, len(xe), 4):
                    xe_sep.extend(xe[idx:idx+4]); xe_sep.append(np.nan)
                    ye_sep.extend(ye[idx:idx+4]); ye_sep.append(np.nan)
                    ze_sep.extend(ze[idx:idx+4]); ze_sep.append(np.nan)

                fig.add_trace(
                    go.Scatter3d(
                        x=xe_sep, y=ye_sep, z=ze_sep,
                        mode="lines",
                        line=dict(color="black", width=0.5),
                        name=f'edges-{hemi}-{v_idx}',
                        showlegend=False
                    ), row=r, col=c
                )

            # Add ROI outlines
            if roi_outlines and rois is not None:
                xe, ye, ze = compute_roi_midline_edges(surf, roi_labels)
                # outline_verts = np.where(outline_mask == 1)[0]

                fig.add_trace(
                    go.Scatter3d(
                        x=xe, y=ye, z=ze,
                        mode="lines",
                        marker=dict(color="black", size=10),
                        name=f'rois-{hemi}-{v_idx}',
                        showlegend=False
                    ), row=r, col=c
                )

            noaxis = dict(showbackground=False, showline=False, zeroline=False, 
                          showgrid=False, showticklabels=False, title="", visible=False)
            # Compute initial aspect ratio
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)
            z_range = np.max(z) - np.min(z)
            max_range = max(x_range, y_range, z_range)
            fig.update_scenes(
                xaxis=noaxis, yaxis=noaxis, zaxis=noaxis,
                aspectmode='manual',
                aspectratio=dict(
                    x=x_range/max_range * zoom,
                    y=y_range/max_range * zoom,
                    z=z_range/max_range * zoom
                ),
                camera=dict(
                    center=dict(x=0, y=0, z=0),
                    eye=camera_views[view]['eye'],
                    up=camera_views[view]['up'],
                    projection=dict(type='orthographic'),
                ),
                row=r, col=c
            )
    
    # General layout
    fig.update_layout(
        width=size[0], height=size[1],
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def fetch_trace(fig, name):
    """
    Fetch a trace from a Plotly figure by its name.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure containing the trace.
    name : str
        The name of the trace to fetch.

    Returns
    -------
    trace : plotly.graph_objects.Trace or None
        The trace with the specified name, or None if not found.
    """
    return next((t for t in fig.data if t.name == name), None)

def update_trace_type(fig, trace_type, **kwargs):
    """
    Update traces by name pattern with given kwargs.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure containing traces to update.
    trace_type : str
        The trace name or prefix to match. Will match traces where name starts with this string.
        Examples: 'data-lh-0' (exact match), 'data' (all data traces).
    **kwargs
        Keyword arguments to pass to trace.update().
    
    """
    for trace in fig.data:
        if trace.name and trace.name.startswith(trace_type):
            trace.update(**kwargs)
    return

def compute_roi_midline_edges(surf, labeling, verbose=False):
    """
    Compute ROI boundaries using midpoints between label boundaries.
    Matches MATLAB findROIboundaries.m behavior, including medial wall borders.
    """
    verts = np.asarray(surf['v'], float)
    tris = np.asarray(surf['t'], int)
    labeling = np.asarray(labeling)
    labeling = np.nan_to_num(labeling, nan=0).astype(int)

    tri_labels = labeling[tris]
    tri_coords = verts[tris]

    line_segments = []
    for lbls, coords in zip(tri_labels, tri_coords):
        unique_lbls = np.unique(lbls)

        # Skip all-zero triangles (pure medial wall)
        if np.all(unique_lbls == 0):
            continue

        edges = [(0, 1), (1, 2), (2, 0)]

        # Two or more distinct labels — boundary triangle
        if len(unique_lbls) == 2:
            # Includes case {0, X}
            diff_edges = [e for e in edges if lbls[e[0]] != lbls[e[1]]]
            if len(diff_edges) == 2:
                mids = [coords[list(e)].mean(axis=0) for e in diff_edges]
                line_segments.append(np.vstack(mids))

        elif len(unique_lbls) == 3:
            # Three-way junction: draw centroid-to-midpoint lines
            centroid = coords.mean(axis=0)
            mids = [coords[list(e)].mean(axis=0) for e in edges]
            for m in mids:
                line_segments.append(np.vstack([centroid, m]))

    if not line_segments:
        if verbose:
            print("No ROI boundaries found.")
        return np.array([]), np.array([]), np.array([])

    segs = np.stack(line_segments)
    n = len(segs)
    xe = np.empty(n * 3)
    ye = np.empty_like(xe)
    ze = np.empty_like(xe)

    xe[0::3] = segs[:, 0, 0]; xe[1::3] = segs[:, 1, 0]; xe[2::3] = np.nan
    ye[0::3] = segs[:, 0, 1]; ye[1::3] = segs[:, 1, 1]; ye[2::3] = np.nan
    ze[0::3] = segs[:, 0, 2]; ze[1::3] = segs[:, 1, 2]; ze[2::3] = np.nan

    return xe, ye, ze


def compute_roi_outlines(surf, labeling):
    """
    Compute a binary mask of border vertices given a parcellation labeling.

    Parameters
    ----------
    surf : dict
        Must contain keys 'v' (vertices, n×3) and 't' (triangles, m×3).
    labeling : array_like of shape (n,)
        ROI label per vertex. Typically integers, with 0 for medial wall.

    Returns
    -------
    border : np.ndarray of shape (n,)
        Binary mask where 1 indicates a vertex on a label boundary.
    """
    tris = np.asarray(surf['t'])
    labeling = np.asarray(labeling)
    n_verts = len(labeling)

    # Get all edges (each as a sorted vertex pair)
    edges = np.sort(
        np.vstack([
            tris[:, [0, 1]],
            tris[:, [1, 2]],
            tris[:, [2, 0]],
        ]),
        axis=1
    )

    # Remove duplicates
    edges = np.unique(edges, axis=0)

    # Find edges with label mismatch
    edge_labels = labeling[edges]
    border_edges = edges[edge_labels[:, 0] != edge_labels[:, 1]]

    # Mark border vertices
    border = np.zeros(n_verts, dtype=np.uint8)
    border[np.unique(border_edges)] = 1

    return border

def fig_to_array(fig, dpi=None, pad_inches=0.0):
    """
    Convert a matplotlib figure to a numpy array.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to convert.
    dpi : float, optional
        Resolution in dots per inch. If None, uses the figure's dpi.
    pad_inches : float, default=0.0
        Amount of padding around the figure when saving.
    
    Returns
    -------
    np.ndarray
        RGB image array with shape (height, width, 3).
    """
    # Save figure to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
    buf.seek(0)
    
    # Load image from buffer and convert to numpy array
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    
    return img_array
