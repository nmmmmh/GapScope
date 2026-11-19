s = open("fig5p.py").read()
head = s[:s.index("# ---------------- (b)")]
c = s[s.index("# ---------------- (c)"):]
old = """    seen = {}
    for _, r in g.sort_values('bandgap_ev').iterrows():
        n = seen.get(round(r.bandgap_ev, 2), 0); seen[round(r.bandgap_ev, 2)] = n + 1
"""
new = """    seen = {}; levels = []   # stack a dot one row lower whenever it sits within 0.06 eV of a dot already on that row
    for _, r in g.sort_values('bandgap_ev').iterrows():
        n = 0
        while n < len(levels) and r.bandgap_ev - levels[n] < 0.06: n += 1
        if n == len(levels): levels.append(r.bandgap_ev)
        else: levels[n] = r.bandgap_ev
        seen[len(seen)] = n
"""
assert old in c; c = c.replace(old, new)
c = c.replace("ax.set_ylim(-0.16 - 0.07 * max(seen.values()), 1.12)", "ax.set_ylim(-0.16 - 0.07 * len(levels), 1.12)")
c = c.replace("fontsize=7, color=TEAL2, fontweight='bold')", "fontsize=7, color=TEAL2)")
open("fig5c_v3.py", "w").write(head + c); print("ok", "fontweight='bold')" in c)
