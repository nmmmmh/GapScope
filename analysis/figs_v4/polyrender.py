"""VESTA-like polyhedral rendering with matplotlib: convex-hull polyhedra, orthographic view, painter's order, Lambert shading."""
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import to_rgb
from pymatgen.core import Structure, Lattice

def rot(view, up=(0, 0, 1)):
    z = np.asarray(view, float); z /= np.linalg.norm(z)
    x = np.cross(up, z); x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.cross([0, 1, 0], z) / np.linalg.norm(np.cross([0, 1, 0], z))
    y = np.cross(z, x); return np.array([x, y, z])

def cluster(s, cation, anion, cut, center_frac, radius, scale=(3, 3, 3)):
    S = s.copy(); S.make_supercell(scale)
    c0 = S.lattice.get_cartesian_coords(center_frac)
    cats = [i for i, x in enumerate(S) if x.specie.symbol == cation and np.linalg.norm(x.coords - c0) < radius]
    polys = []
    for i in cats:
        nb = S.get_neighbors(S[i], cut)
        pts = np.array([n.coords for n in nb if n.specie.symbol == anion])
        if len(pts) >= 4: polys.append((S[i].coords, pts))
    return polys, c0

def draw(ax, polys, c0, view, colors, light=(-0.4, 0.6, 0.7), edge='#2b3a40', o_col='#E2463A', o_r=0.32, lw=0.35, shift=(0, 0), scale=1.0, up=(0, 0, 1)):
    R = rot(view, up); L = np.asarray(light, float); L /= np.linalg.norm(L)
    faces, atoms = [], {}
    for cen, pts in polys:
        cn = len(pts); base = np.array(to_rgb(colors.get(cn, colors['default'])))
        h = ConvexHull(pts)
        for simp in h.simplices:
            tri = pts[simp]; n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); n /= np.linalg.norm(n)
            if np.dot(n, tri.mean(0) - cen) < 0: n = -n
            nv = R @ n
            if nv[2] < -0.02: continue                          # back faces hidden
            p = (R @ (tri - c0).T).T
            shade = 0.58 + 0.42 * max(0.0, float(np.dot(R.T @ L, n)))
            faces.append((p[:, 2].mean(), p[:, :2] * scale + shift, np.clip(base * shade + (1 - shade) * 0.06, 0, 1)))
        for q in pts: atoms[tuple(np.round(q, 3))] = q
    items = [(d, 'f', xy, c) for d, xy, c in faces]
    for q in atoms.values():
        p = R @ (q - c0); items.append((p[2] + 0.05, 'o', p[:2] * scale + shift, None))
    for d, kind, xy, c in sorted(items, key=lambda t: t[0]):
        if kind == 'f':
            ax.add_patch(Polygon(xy, closed=True, fc=c, ec=edge, lw=lw, joinstyle='round', alpha=0.96))
        else:
            ax.add_patch(Circle(xy, o_r * scale, fc=o_col, ec='#8E2219', lw=0.3, zorder=None))
            ax.add_patch(Circle(xy + np.array([-0.32, 0.34]) * o_r * scale, 0.36 * o_r * scale, fc='white', ec='none', alpha=0.55))
