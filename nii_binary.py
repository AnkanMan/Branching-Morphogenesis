# save this as convert_oib_to_nifti.py

import oiffile, nibabel as nib, numpy as np, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='.oib file path')
parser.add_argument('-o', '--output', help='Output .nii.gz path')
args = parser.parse_args()

with oiffile.OifFile(args.input) as oib:
    data = oib.asarray()
    data = np.squeeze(data)
    z = oib.mainfile['Axis 3 Parameters Common']
    y = oib.mainfile['Axis 1 Parameters Common']
    x = oib.mainfile['Axis 0 Parameters Common']
    spacing = [(float(z['EndPosition']) / data.shape[0]),
               (float(y['EndPosition']) / data.shape[1]),
               (float(x['EndPosition']) / data.shape[2])]

if args.output:
    out_path = args.output
else:
    out_path = os.path.splitext(args.input)[0] + ".nii.gz"

affine = np.diag(spacing + [1.0])
nib.save(nib.Nifti1Image(data.astype(np.uint16), affine), out_path)
print(f"Saved: {out_path}")
