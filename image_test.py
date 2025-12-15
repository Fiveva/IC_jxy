import tifffile
try:
    data = tifffile.imread('whole_brain_3d.tif')
    print("Tifffile successfully read the file. Shape:", data.shape)
except Exception as e:
    print("Tifffile failed:", e)