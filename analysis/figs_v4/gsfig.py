"""Shared style and helpers for GapScope figures v4 (JCTC, 7.0 in double column)."""
import re, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyBboxPatch
from scipy.spatial import ConvexHull

FONT = 'Liberation Sans'
plt.rcParams.update({
    'font.family': FONT, 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out', 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 6.5, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': FONT + ':italic',
    'mathtext.bf': FONT + ':bold', 'axes.labelpad': 2.5, 'xtick.major.pad': 1.8, 'ytick.major.pad': 1.8,
    'lines.linewidth': 0.9, 'savefig.dpi': 600})

INK = '#1f2a44'; GREY = '#8d939b'; LGREY = '#d5d9de'; XLGREY = '#eceef1'; ACC = '#2b5d9b'
PH_C = ['#D55E00', '#0072B2', '#009E73', '#CC79A7']   # phase 1-4 within a composition (Okabe-Ito)
PH_M = ['*', 'D', '^', 's']
PH_S = [5.0, 2.6, 3.0, 2.6]                            # marker sizes that look equal
VESTA = {'Pb': '#575961', 'I': '#940094', 'Ga': '#c2c48f', 'O': '#fe0300', 'Si': '#1b3bfa', 'C': '#814929',
         'Bi': '#d22ff7', 'Ti': '#78caff', 'Se': '#9aef0f', 'In': '#a67573'}

def fmt(formula):
    """Ga2O3 -> Ga$_2$O$_3$ (mathtext subscripts)."""
    return re.sub(r'(\d+\.?\d*)', r'$_{\1}$', formula)

def label(ax, s, x=-0.02, y=1.02, fig=None):
    ax.text(x, y, '(%s)' % s, transform=ax.transAxes, fontsize=8, fontweight='bold', ha='right', va='bottom')

def flabel(fig, s, x, y):
    fig.text(x, y, '(%s)' % s, fontsize=8, fontweight='bold', ha='left', va='top')

# ---------------------------------------------------------------- structure renderer (VESTA-like, orthographic)
def view(az, el):
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz

def _shade(hexc, f):
    c = np.array(matplotlib.colors.to_rgb(hexc)); return tuple(np.clip(c * f, 0, 1)) if f < 1 else tuple(c + (1 - c) * (f - 1))

def render(ax, polys, atoms, R, poly_color, poly_alpha=0.55, edge_lw=0.35, light=np.array([-0.4, 0.5, 0.75])):
    """polys: list of (N,3) vertex arrays (convex); atoms: list of (pos, element, radius)."""
    light = light / np.linalg.norm(light); items = []
    for V in polys:
        P = V @ R.T; hull = ConvexHull(V)
        cen = V.mean(0)
        for s in hull.simplices:
            tri = P[s]; n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); n /= np.linalg.norm(n) + 1e-12
            if np.dot(n, (V[s].mean(0) - cen) @ R.T) < 0: n = -n
            b = 0.78 + 0.32 * max(0.0, float(np.dot(n, light)))
            items.append((tri[:, 2].mean(), 'f', tri[:, :2], b))
    for pos, el, r in atoms:
        p = np.asarray(pos) @ R.T; items.append((p[2] + 1e-3, 'a', p[:2], (el, r)))
    for z, kind, xy, info in sorted(items, key=lambda t: t[0]):
        if kind == 'f':
            ax.add_patch(Polygon(xy, closed=True, fc=_shade(poly_color, info), ec=_shade(poly_color, 0.55),
                                 lw=edge_lw, alpha=poly_alpha, joinstyle='round'))
        else:
            el, r = info; c = VESTA.get(el, '#999999')
            ax.add_patch(Circle(xy, r, fc=c, ec=_shade(c, 0.55), lw=0.3, zorder=None))
            ax.add_patch(Circle(xy + np.array([-0.3, 0.32]) * r, 0.38 * r, fc='white', ec='none', alpha=0.45))
    ax.set_aspect('equal'); ax.axis('off'); ax.autoscale_view()

def octahedron(c, s=1.0):
    d = s / np.sqrt(2); c = np.asarray(c, float)
    return np.array([c + v for v in ([d, 0, 0], [-d, 0, 0], [0, d, 0], [0, -d, 0], [0, 0, d], [0, 0, -d])])

def corner_sharing(n=(2, 2, 2), a=1.0):
    polys, atoms, seen = [], [], set()
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                c = np.array([i, j, k], float) * a; V = octahedron(c, a / np.sqrt(2))
                polys.append(V)
                for v in V:
                    key = tuple(np.round(v, 3))
                    if key not in seen: seen.add(key); atoms.append((v, 'I', 0.12 * a))
    return polys, atoms

def face_sharing_chain(xy, n=4, s=1.0, phase=0.0):
    r, h = s / np.sqrt(3), s / np.sqrt(6)
    polys, atoms, seen = [], [], set()
    for k in range(n):
        phi = phase - 60.0 * k; zc = 2 * h * k
        top = [(xy[0] + r * np.cos(np.radians(phi + t)), xy[1] + r * np.sin(np.radians(phi + t)), zc + h) for t in (0, 120, 240)]
        bot = [(xy[0] + r * np.cos(np.radians(phi + 60 + t)), xy[1] + r * np.sin(np.radians(phi + 60 + t)), zc - h) for t in (0, 120, 240)]
        V = np.array(top + bot); polys.append(V)
        for v in V:
            key = tuple(np.round(v, 3))
            if key not in seen: seen.add(key); atoms.append((v, 'I', 0.12 * s))
    return polys, atoms

def save(fig, path):
    fig.savefig(path + '.pdf'); fig.savefig(path + '.png', dpi=300)
