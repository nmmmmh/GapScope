"""Ray-traced (Tachyon, ambient occlusion) polyhedral pictures of alpha- and beta-Ga2O3 for the TOC."""
import json, sys, numpy as np
from pymatgen.core import Structure, Lattice
from ovito.data import DataCollection, ParticleType
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import CreateBondsModifier, ExpressionSelectionModifier, CoordinationPolyhedraModifier
from ovito.vis import Viewport, TachyonRenderer
d = json.load(open('/data/czx/GapScope_Revision_20261113/struct/sel_structures.json'))
def S(k): v = d[k]; return Structure(Lattice(v['lattice']), v.get('species', v.get('elements')), v['coords'])
OCT, TET, OX = (0.48, 0.75, 0.71), (0.12, 0.44, 0.41), (0.86, 0.27, 0.22)   # octahedra light teal, tetrahedra dark teal, O red
def cluster(s, radius, scale=(4, 4, 4)):
    s = s.copy(); s.make_supercell(scale); c0 = s.lattice.get_cartesian_coords((0.5, 0.5, 0.5))
    ga = [i for i, x in enumerate(s) if x.specie.symbol == 'Ga' and np.linalg.norm(x.coords - c0) < radius]
    pos, typ, col, oset = [], [], [], {}
    for i in ga:
        nb = [n for n in s.get_neighbors(s[i], 2.3) if n.specie.symbol == 'O']
        pos.append(s[i].coords); typ.append(1); col.append(OCT if len(nb) >= 6 else TET)
        for n in nb: oset[tuple(np.round(n.coords, 3))] = n.coords
    for q in oset.values(): pos.append(q); typ.append(2); col.append(OX)
    return np.array(pos) - c0, np.array(typ), np.array(col)
def render(key, radius, cam, out, size=1400):
    pos, typ, col = cluster(S(key), radius)
    data = DataCollection()
    cell = data.create_cell(np.hstack([np.eye(3) * 60, [[-30], [-30], [-30]]]), pbc=(False, False, False)); cell.vis.enabled = False
    p = data.create_particles(count=len(pos)); p.create_property('Position', data=pos)
    t = p.create_property('Particle Type', data=typ)
    t.types.append(ParticleType(id=1, name='Ga', color=OCT, radius=0.05))
    t.types.append(ParticleType(id=2, name='O', color=OX, radius=0.36))
    p.create_property('Color', data=col)
    pl = Pipeline(source=StaticSource(data=data))
    cb = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise); cb.set_pairwise_cutoff('Ga', 'O', 2.3); cb.vis.enabled = False
    pl.modifiers.append(cb)
    pl.modifiers.append(ExpressionSelectionModifier(expression='ParticleType==1'))
    cp = CoordinationPolyhedraModifier(); cp.vis.highlight_edges = True; cp.vis.surface_transparency = 0.0
    pl.modifiers.append(cp)
    pl.add_to_scene()
    vp = Viewport(type=Viewport.Type.Ortho, camera_dir=cam); vp.zoom_all()
    r = TachyonRenderer(shadows=False, ambient_occlusion=True, ambient_occlusion_brightness=0.85, antialiasing_samples=16, direct_light_intensity=0.95)
    vp.render_image(size=(size, size), filename=out, background=(1, 1, 1), alpha=True, renderer=r)
    pl.remove_from_scene(); print(out, len(pos))
render('SM-19471', 6.0, (-1.0, -0.35, -0.25), 'ovito_alpha.png')
render('SM-19470', 6.0, (-0.15, -1.0, -0.30), 'ovito_beta.png')
