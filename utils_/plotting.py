import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation


def plot_image_grid(images, grid_dims=[3, 3]):
    fig, axes = plt.subplots(grid_dims[1], grid_dims[0], figsize=(10, 10))
    for i in range(grid_dims[1]):
        for j in range(grid_dims[0]):
            axes[i, j].imshow(images[i*grid_dims[0]+j])
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

def plot_path(x: np.ndarray, y: np.ndarray, markings_x=[], markings_y=[]):
    # Create a color gradient from blue to red based on the index
    cmap = LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])
    norm = plt.Normalize(0, len(x) - 1)  # Normalize based on index

    # Prepare the segments for the gradient line
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create an array of indices for color mapping
    indices = np.linspace(0, len(x) - 1, len(x))

    # Create the LineCollection with the gradient
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(indices)  # Use the indices for color mapping
    lc.set_linewidth(2)

    # Plot the gradient line
    fig, ax = plt.subplots()
    ax.add_collection(lc)

    # Add colorbar for reference
    cbar = plt.colorbar(lc, ax=ax, orientation='horizontal', label='Index')
    cbar.set_label('Index', rotation=0)

    # Set limits for the axes
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    ax.axis("equal")

    # Custom legend
    legend_elements = [Line2D([0], [0], color='blue', lw=4, label='Start (Index 0)'),
                    Line2D([0], [0], color='red', lw=4, label='End (Index {})'.format(len(x) - 1))]
    ax.legend(handles=legend_elements, loc='upper right')

    if len(markings_x):
        ax.scatter(markings_x, markings_y, c="g", s=200)

    plt.show()


def vis_denoising(paths):
    paths = paths.transpose(0, 2, 1)
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize the scatter plot
    sc = ax.scatter([], [], [], c='r', marker='o')

    # Set up the axis limits
    xyzs = np.concatenate(paths, axis=1)
    ax.set_xlim(min(xyzs[0]), max(xyzs[0]))
    ax.set_ylim(min(xyzs[1]), max(xyzs[1]))
    ax.set_zlim(min(xyzs[2]), max(xyzs[2]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Animated 3D Scatter Plot')

    # Initialize function to set up the background of each frame
    def init():
        sc._offsets3d = ([], [], [])  # clear the data
        return sc,

    # Animation function which is called sequentially
    def animate(i):
        x, y, z = paths[i]
        sc._offsets3d = (x, y, z)
        return sc,

    # Create the animation object
    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(paths), interval=10, blit=False
    )

    # Show the animation
    plt.show()