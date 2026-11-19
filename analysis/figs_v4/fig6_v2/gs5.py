"""Style v5: matches the author's PPT schematics (Arial, PPT palette). Vector output."""
import os, matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager as fm
D = os.path.dirname(os.path.abspath(__file__))
for f in ('arial.ttf', 'arialbd.ttf', 'ariali.ttf'):
    if os.path.exists(os.path.join(D, f)): fm.fontManager.addfont(os.path.join(D, f))
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 6.5,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'axes.labelpad': 2.5, 'xtick.major.pad': 1.8,
    'ytick.major.pad': 1.8, 'axes.edgecolor': '#123B5C', 'xtick.color': '#123B5C', 'ytick.color': '#123B5C',
    'axes.labelcolor': '#000000', 'text.color': '#000000', 'savefig.dpi': 600})
TEAL, TEAL2, BLUE, PURP, NAVY, AMBER, SLATE = '#158E91', '#08797B', '#2979C9', '#8559BD', '#123B5C', '#C27A28', '#708596'
GREYP, LTEAL, LBLUE = '#A9B3BD', '#E0F2F1', '#CCDBE9'
import re
def fmt(f): return re.sub(r'(\d+\.?\d*)', r'$_{\1}$', f)

# ---- tiered palette (v2): hero = saturated teal; secondary = light steel blue; context = cool grey; accent = amber (callouts only)
TEAL, TEAL2, LTEAL = '#139088', '#08605E', '#D3EBE8'          # hero main / hero dark / hero tint
BLUE, LBLUE = '#8DB4DC', '#E1ECF7'                            # secondary mid / tint
PURP = '#A396CF'                                              # muted violet (tertiary)
GREYP, SLATE = '#BCC5CE', '#7D8A98'                           # context
NAVY, AMBER = '#123B5C', '#C27A28'
PH = ['#139088', '#3E7CB8', '#7B66B3', '#C27A28']             # phases within a composition, equal weight

# ---- palette v3: warm-cool layering. cool teal family carries data and models; warm coral marks the key quantity
# (phase effect, floor, variable coordination, matched residual); sand is the single callout; charcoal-navy is ink.
TEAL, TEAL2, LTEAL = '#2A9D8F', '#1F6F68', '#D5EDE9'
CORAL, LCORAL = '#E76F51', '#FBE3DB'
SAND, LSAND = '#E9A23B', '#F8E7C2'
INDIGO = '#457B9D'
BLUE, LBLUE = '#8FB3CC', '#E3ECF2'
PURP = '#9C89C9'
GREYP, SLATE = '#C3CDD3', '#7F929C'
NAVY = '#264653'
AMBER = SAND
PH = ['#2A9D8F', '#457B9D', '#E76F51', '#E9A23B']
