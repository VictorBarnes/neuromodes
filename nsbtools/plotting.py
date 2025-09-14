import tempfile
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1 import make_axes_locatable
from surfplot import Plot

def plot_brain(surf, data, layout="row", views=["lateral", "medial"], color_range="individual", 
               cmap="viridis", cbar=False, cbar_label=None, cbar_kws=None, labels=None, 
               label_kws=None, outline=False, zoom=1.25, ax=None):
    """
    Plot brain surface data on a given surface mesh.

    Parameters
    ----------
    surf : str
        Path to the surface file.
    data : array-like
        Data to be plotted on the surface. Can be 1D or 2D.
    layout : str, optional
        Layout of the subplots, either "row" or "col", by default "row".
    views : list of str, optional
        List of views to display, by default ["lateral", "medial"].
    color_range : tuple of float, str, or None, optional
        Defines the color limits for the colormap. Can be:
        - A tuple (vmin, vmax) to apply the same color scale across all maps.
        - "group" to compute global (min, max) across all data columns and apply uniformly.
        - "individual" (or None) to compute limits separately for each brain map.
        - "individual_centered" to compute limits centered around zero for each brain map.
        By default, color range is determined individually per map.
    cmap : matplotlib colormap name or object, optional
        Colormap to use for the data, by default "viridis".
    cbar : bool, optional
        Whether to display a colorbar, by default False.
    cbar_label : str, optional
        Label for the colorbar, by default None.
    cbar_kws : dict, optional
        Additional keyword arguments for the colorbar, by default None.
    labels : list of str, optional
        List of labels for each subplot, by default None.
    label_kws : dict, optional
        Additional keyword arguments for the labels, by default None.
    outline : bool, optional
        Whether to outline the data, by default False. Useful for parcellations.
    zoom : float, optional
        Zoom factor for the brain plot, by default 1.25.
    ax : matplotlib.axes.Axes or list of Axes, optional
        Axis or list of axes to plot on. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure or None
        The resulting figure if a new one is created, otherwise None.
    """
    
    # Validate inputs
    cbar_kws_ = {**dict(pad=0.01, fontsize=20, aspect=25, shrink=1, decimals=2, location="bottom"),
                 **(cbar_kws or {})}
    label_kws_ = {**dict(fontsize=20), **(label_kws or {})}
    
    data = np.squeeze(data)
    if np.ndim(data) == 1 or np.shape(data)[1] == 1:
        data = data.reshape(-1, 1)
    
    n_data = np.shape(data)[1]
    
    # Create the figure and axes
    if ax is None:
        if layout == "row":
            fig, axs = plt.subplots(1, n_data, figsize=(len(views) * n_data * 1.5, 2))
        elif layout == "col":
            fig, axs = plt.subplots(n_data, 1, figsize=(3, n_data * 1.25))
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        axs = [axs] if n_data == 1 else axs.flatten()
    else:
        if isinstance(ax, list):
            if len(ax) != n_data:
                raise ValueError("Number of provided axes must match the number of brains to plot.")
            axs = ax
        else:
            if n_data > 1:
                raise ValueError("Multiple brains require a list of axes.")
            axs = [ax]
    
    # Set the color range
    if isinstance(color_range, str) and color_range == "group":
        color_range = (np.nanmin(data), np.nanmax(data))
    elif isinstance(color_range, str) and color_range == "individual":
        color_range = None

    # To plot multiple brain maps, save each figure to a temporary file then load it into the axes
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, ax in enumerate(axs):
            if isinstance(color_range, str) and color_range == "individual_centered":
                max_abs = np.nanmax(np.abs(data[:, i]))
                color_range = (-max_abs, max_abs)

            # Use surfplot to plot the data
            p = Plot(surf_lh=surf, views=views, size=(500, 250), zoom=zoom)
            p.add_layer(data=data[:, i], cmap=cmap, cbar=cbar, color_range=color_range,
                        cbar_label=cbar_label, zero_transparent=False)
            if outline:
                p.add_layer(data[:, i], as_outline=True, cmap="gray", cbar=False,
                            color_range=(1, 2), zero_transparent=False)
            temp_file = f"{temp_dir}/figure_{i}.png"
            fig = p.build(cbar_kws=cbar_kws_)
            plt.close(fig)
            
            # Save the surfplot figure
            fig.savefig(temp_file, bbox_inches='tight')
            # Load the figure into the axes
            ax.imshow(plt.imread(temp_file))
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            # Plot labels
            if labels is not None:
                if layout == "row":
                    ax.set_title(labels[i], pad=0, fontsize=label_kws_["fontsize"])
                elif layout == "col":
                    ax.set_ylabel(labels[i], labelpad=0, rotation=0, ha="right",
                                  fontsize=label_kws_["fontsize"])
    
    return fig if ax is None else None

def plot_heatmap(data, ax=None, center=None, cmap="viridis", cbar=False, square=True,
                 downsample=1, annot=False, fmt=".1f"):
    """
    Plot a heatmap of the data with optional colorbar and annotations.

    Parameters
    ----------
    data : 2D array-like
        The data to be plotted as a heatmap.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the heatmap. If None, the current axes are used.
    center : float, optional
        Center value for colormap scaling. If None, the colormap is not centered.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to be used for the heatmap. Default is "viridis".
    cbar : bool, optional
        Whether to display a colorbar beside the heatmap. Default is False.
    square : bool, optional
        If True, set the aspect ratio of the plot to be equal, so the cells are square-shaped.
        Default is True.
    downsample : float, optional
        Factor by which to downsample the data before plotting. Should be between 0 and 1.
    annot : bool, optional
        Whether to annotate each cell with its value. Default is False.
    fmt : str, optional
        String format for the annotations. Default is ".1f" (one decimal place).

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the heatmap is plotted.
    """

    if ax is None:
        ax = plt.gca()

    if 0 < downsample < 1:
        data = zoom(data, zoom=downsample, order=1) # bilinear interpolation

    vmin = np.min(data)
    vmax = np.max(data)

    cmap = plt.get_cmap(cmap)
    if center is not None:
        # Compute a symmetric range around center
        vrange = max(vmax - center, center - vmin)
        norm = mpl.colors.Normalize(vmin=center - vrange, vmax=center + vrange)

        # Remap the colormap to ensure center=0 is white
        cmin, cmax = norm([vmin, vmax])
        cc = np.linspace(cmin, cmax, 256)
        cmap = mpl.colors.ListedColormap(cmap(cc))

    # Plot heatmap with colorbar
    mesh = ax.pcolormesh(data, cmap=cmap, **{"vmin": vmin, "vmax": vmax})

    # Invert the y axis to show the plot in matrix form
    ax.invert_yaxis()

    # Annotate each cell with its value
    if annot:
        norm = mesh.norm  # get the normalization used by pcolormesh
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Use black or white text depending on background luminance
                val = data[i, j]
                r, g, b = cmap(norm(val))[:3]
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = 'white' if luminance < 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, format(val, fmt), ha='center', va='center',
                        color=text_color)

    # Create a colorbar with the same height as the heatmap
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)  # Adjust size and padding
        cb = plt.colorbar(mesh, cax=cax)

    # Set frame around heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    ax.set_xticks([])
    ax.set_yticks([])
    if square:
        ax.set_aspect("equal")

    return ax