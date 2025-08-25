import napari
import nibabel as nib

# Load NIfTI data using nibabel
nii = nib.load("E18.5-4x-brain-tile-1.nii.gz")
#nii = nib.load("output_file2.nii.gz")
data = nii.get_fdata()

# Optional: normalize contrast range
vmin, vmax = data.min(), data.max()

# Launch viewer
viewer = napari.Viewer()
viewer.add_image(data, name="Vessel Data", colormap='gray', contrast_limits=[vmin, vmax])
napari.run()
#conda install -c conda-forge pyqt
#conda install -c conda-forge pyside2