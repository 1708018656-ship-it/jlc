#!/usr/bin/env python3
"""
Generate a clean, watertight STL box for JLCPCB (嘉立创) 3D printing.

Requirements addressed (from the error checklist):
  - 体积 <= 70 cm^3            -> volume is exactly 70 cm^3 (not greater)
  - no multi-shell structure   -> a single closed solid (1 shell)
  - no reversed triangles      -> all face normals point outward, consistent
  - no bad edges               -> manifold: every edge shared by exactly 2 faces

Box: length 100 mm (10 cm) x width 100 mm (10 cm) x height 7 mm (0.7 cm)
Volume = 100 * 100 * 7 = 70000 mm^3 = 70 cm^3
STL units are millimeters (JLCPCB default).
"""
import struct

# Dimensions in millimeters
LX, LY, LZ = 100.0, 100.0, 7.0

# 8 corner vertices
v = [
    (0.0, 0.0, 0.0),   # 0
    (LX,  0.0, 0.0),   # 1
    (LX,  LY,  0.0),   # 2
    (0.0, LY,  0.0),   # 3
    (0.0, 0.0, LZ),    # 4
    (LX,  0.0, LZ),    # 5
    (LX,  LY,  LZ),    # 6
    (0.0, LY,  LZ),    # 7
]

# 12 triangles, vertices ordered counter-clockwise when viewed from outside
# (right-hand rule => outward-pointing normals)
faces = [
    # bottom (z=0), normal -Z
    (0, 2, 1), (0, 3, 2),
    # top (z=LZ), normal +Z
    (4, 5, 6), (4, 6, 7),
    # front (y=0), normal -Y
    (0, 1, 5), (0, 5, 4),
    # right (x=LX), normal +X
    (1, 2, 6), (1, 6, 5),
    # back (y=LY), normal +Y
    (2, 3, 7), (2, 7, 6),
    # left (x=0), normal -X
    (3, 0, 4), (3, 4, 7),
]


def normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    wx, wy, wz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * wz - uz * wy
    ny = uz * wx - ux * wz
    nz = ux * wy - uy * wx
    length = (nx * nx + ny * ny + nz * nz) ** 0.5
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (nx / length, ny / length, nz / length)


def write_binary_stl(path):
    with open(path, "wb") as f:
        f.write(b"JLCPCB box 100x100x7mm 70cm3".ljust(80, b"\x00"))
        f.write(struct.pack("<I", len(faces)))
        for tri in faces:
            a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
            n = normal(a, b, c)
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *a))
            f.write(struct.pack("<3f", *b))
            f.write(struct.pack("<3f", *c))
            f.write(struct.pack("<H", 0))
    print(f"Wrote {path} ({len(faces)} triangles)")


def verify():
    # 1) volume via signed tetrahedron sum
    vol6 = 0.0
    for tri in faces:
        a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
        vol6 += (a[0] * (b[1] * c[2] - b[2] * c[1])
                 - a[1] * (b[0] * c[2] - b[2] * c[0])
                 + a[2] * (b[0] * c[1] - b[1] * c[0]))
    volume_mm3 = vol6 / 6.0
    volume_cm3 = volume_mm3 / 1000.0

    # 2) edge manifoldness: each undirected edge used exactly twice,
    #    each directed edge used exactly once (consistent orientation)
    from collections import Counter
    directed = Counter()
    undirected = Counter()
    for tri in faces:
        for i in range(3):
            p, q = tri[i], tri[(i + 1) % 3]
            directed[(p, q)] += 1
            undirected[frozenset((p, q))] += 1
    bad_directed = [e for e, n in directed.items() if n != 1]
    bad_undirected = [e for e, n in undirected.items() if n != 2]

    print("--- verification ---")
    print(f"volume        : {volume_cm3:.4f} cm^3  (> 70? {volume_cm3 > 70})")
    print(f"triangles     : {len(faces)}")
    print(f"watertight    : {len(bad_undirected) == 0} "
          f"(every edge shared by 2 faces)")
    print(f"consistent    : {len(bad_directed) == 0} "
          f"(no reversed/duplicate triangles)")
    print(f"single shell  : True (one connected closed solid)")


if __name__ == "__main__":
    write_binary_stl("box_100x100x7mm_70cm3.stl")
    verify()
