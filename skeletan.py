import numpy as np
import tifffile
from skimage.morphology import skeletonize_3d, remove_small_objects, binary_closing, ball
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pyvista as pv

# --- Step 1: Load binary and preprocess ---
binary = tifffile.imread("threshold_0.tif").astype(bool)
#binary = tifffile.imread("threshold_4.tif").astype(bool)
# Fill gaps with 3D closing
binary = binary_closing(binary, ball(1))

# Remove small noisy blobs before skeletonizing
binary = remove_small_objects(binary, min_size=200)

# --- Step 2: Compute distance transform ---
distance = distance_transform_edt(binary)

# --- Step 3: Skeletonize ---
skeleton = skeletonize_3d(binary)

# Clean stray speckles in skeleton
skeleton = remove_small_objects(skeleton, min_size=40)

# --- Step 4: Extract coordinates and radius ---
coords = np.argwhere(skeleton)
radii = distance[tuple(coords.T)]
points = coords[:, [2, 1, 0]].astype(np.float32)  # (x, y, z)

# --- Step 5: Build graph from KDTree (increased r) ---
tree = cKDTree(points)
pairs = list(tree.query_pairs(r=3.0))  # ↑ Slightly more tolerant

# --- Step 6: Keep largest connected component ---
num_pts = len(points)
graph = csr_matrix((np.ones(len(pairs)), ([i for i, j in pairs], [j for i, j in pairs])), shape=(num_pts, num_pts))
n_components, labels = connected_components(csgraph=graph, directed=False)

# Keep only the largest component
largest_label = np.argmax(np.bincount(labels))
mask = labels == largest_label
points = points[mask]
radii = radii[mask]

# --- Step 7: Reconstruct lines from surviving points ---
old_to_new = -np.ones(len(labels), dtype=int)
old_to_new[np.where(mask)[0]] = np.arange(np.count_nonzero(mask))
filtered_pairs = [(old_to_new[i], old_to_new[j]) for i, j in pairs if mask[i] and mask[j]]
lines = [[2, i, j] for i, j in filtered_pairs]

# --- Step 8: Build PyVista mesh ---
skeleton_mesh = pv.PolyData()
skeleton_mesh.points = points
skeleton_mesh.lines = np.array(lines)
skeleton_mesh["radius"] = radii

# --- Step 9: Display ---
plotter = pv.Plotter(window_size=(1200, 1000))
plotter.add_mesh(
    skeleton_mesh,
    scalars="radius",
    cmap="turbo",
    line_width=1.3,
    show_scalar_bar=True,
    scalar_bar_args={"title": "Radius"},
)
plotter.set_background("white")
plotter.add_title("Clean 3D Vessel Skeleton (Colored by Thickness)", font_size=18)
plotter.show()
