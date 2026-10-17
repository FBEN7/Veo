import sys, zipfile, io, pickletools, numpy as np
from numpy.lib import format as fmt
D = sys.argv[1]
zf = zipfile.ZipFile(f"{D}/117093_camera_intrinsics.npz")
for n in zf.namelist():
    b = zf.read(n); f = io.BytesIO(b)
    ver = fmt.read_magic(f)
    shape, fo, dt = (fmt.read_array_header_1_0 if ver==(1,0) else fmt.read_array_header_2_0)(f)
    print(n, shape, dt)
    if dt == object:
        rest = f.read()
        pickletools.dis(rest)
z = np.load(f"{D}/117093_camera_intrinsics.npz", allow_pickle=False)
print(z.files)
for n in ("mapx", "mapy"):
    m = np.load(f"{D}/117093_{n}.npy", mmap_mode="r", allow_pickle=False)
    print(n, m.dtype, m.shape, float(np.nanmin(m)), float(np.nanmax(m)))
    print(" corners", m[0,0], m[0,-1], m[-1,0], m[-1,-1], " centre", m[m.shape[0]//2, m.shape[1]//2])
