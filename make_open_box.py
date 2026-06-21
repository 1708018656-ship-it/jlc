#!/usr/bin/env python3
"""
Generate a clean, watertight HOLLOW box (open-top container) STL for
JLCPCB (嘉立创) 3D printing.

Shape: a box that is hollow inside (a tray/container), open at the top.
  - footprint (outer): 100 mm x 100 mm  (10 cm x 10 cm)
  - height (outer)    : H mm            (adjustable; not fixed by user)
  - wall thickness    : T mm
  - bottom thickness  : B mm

The value JLCPCB checks ("体积") is the *material* volume = walls + floor,
NOT the bounding-box volume. We keep material volume <= 70 cm^3.

Rules satisfied (the error checklist):
  - 体积 <= 70 cm^3            -> material volume printed below, kept under 70
  - no multi-shell structure   -> one single connected, drainable surface
  - no reversed triangles      -> normals made consistent & outward
  - no bad edges               -> manifold: every edge shared by exactly 2 faces
STL units are millimeters (JLCPCB default).
"""
import numpy as np
import trimesh

# ---- parameters (mm) ----
OUT = 100.0   # outer footprint (length = width = 100 mm = 10 cm)
H = 50.0      # outer height (5 cm) -- adjustable
T = 2.0       # wall thickness
B = 2.0       # bottom (floor) thickness

inn = OUT - 2 * T  # inner cavity footprint side

# ---- vertices ----
# Outer bottom (z=0)
O0 = (0.0,  0.0,  0.0); O1 = (OUT, 0.0,  0.0); O2 = (OUT, OUT, 0.0); O3 = (0.0, OUT, 0.0)
# Outer top (z=H)
O4 = (0.0,  0.0,  H);   O5 = (OUT, 0.0,  H);   O6 = (OUT, OUT, H);   O7 = (0.0, OUT, H)
# Inner top (z=H), inset by T
I4 = (T,    T,    H);   I5 = (OUT-T, T,  H);   I6 = (OUT-T, OUT-T, H); I7 = (T, OUT-T, H)
# Inner floor (z=B)
I0 = (T,    T,    B);   I1 = (OUT-T, T,  B);   I2 = (OUT-T, OUT-T, B); I3 = (T, OUT-T, B)

verts = [O0, O1, O2, O3, O4, O5, O6, O7, I4, I5, I6, I7, I0, I1, I2, I3]
idx = {id(v): i for i, v in enumerate(verts)}
# index helpers
o0, o1, o2, o3, o4, o5, o6, o7 = 0, 1, 2, 3, 4, 5, 6, 7
i4, i5, i6, i7 = 8, 9, 10, 11
i0, i1, i2, i3 = 12, 13, 14, 15


def quad(a, b, c, d):
    return [(a, b, c), (a, c, d)]


faces = []
# Outer bottom (z=0)
faces += quad(o0, o1, o2, o3)
# Outer side walls
faces += quad(o0, o1, o5, o4)   # front y=0
faces += quad(o1, o2, o6, o5)   # right x=OUT
faces += quad(o2, o3, o7, o6)   # back y=OUT
faces += quad(o3, o0, o4, o7)   # left x=0
# Top rim (annular frame at z=H, between outer top and inner top)
faces += quad(o4, o5, i5, i4)
faces += quad(o5, o6, i6, i5)
faces += quad(o6, o7, i7, i6)
faces += quad(o7, o4, i4, i7)
# Inner side walls (cavity)
faces += quad(i4, i5, i1, i0)
faces += quad(i5, i6, i2, i1)
faces += quad(i6, i7, i3, i2)
faces += quad(i7, i4, i0, i3)
# Inner floor (cavity bottom, z=B)
faces += quad(i0, i1, i2, i3)

mesh = trimesh.Trimesh(vertices=np.array(verts, dtype=float),
                       faces=np.array(faces, dtype=int),
                       process=True)
# Make winding/normals consistent and outward-facing (fixes reversed triangles)
mesh.fix_normals()

out_path = f"open_box_{int(OUT/10)}x{int(OUT/10)}cm_h{int(H/10) if H%10==0 else H/10}cm.stl"
out_path = "open_box_10x10cm.stl"
mesh.export(out_path)

# ---- report / verification ----
analytic = OUT * OUT * H - inn * inn * (H - B)  # material volume (mm^3)
print(f"wrote: {out_path}")
print("--- geometry ---")
print(f"outer footprint : {OUT} x {OUT} mm  ({OUT/10:.0f} x {OUT/10:.0f} cm)")
print(f"outer height    : {H} mm  ({H/10:.1f} cm)")
print(f"wall thickness  : {T} mm,  bottom thickness: {B} mm")
print(f"inner cavity    : {inn} x {inn} x {H-B} mm  (open top)")
print("--- verification (trimesh) ---")
print(f"material volume : {mesh.volume/1000.0:.3f} cm^3   (analytic {analytic/1000.0:.3f})")
print(f"  -> <= 70 cm^3 : {mesh.volume/1000.0 <= 70.0}")
print(f"triangles       : {len(mesh.faces)},  vertices: {len(mesh.vertices)}")
print(f"watertight      : {mesh.is_watertight}   (no bad edges)")
print(f"winding ok      : {mesh.is_winding_consistent}   (no reversed faces)")
print(f"single shell    : {mesh.body_count == 1}   (body_count={mesh.body_count})")
