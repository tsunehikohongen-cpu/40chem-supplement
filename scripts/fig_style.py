"""Shared matplotlib style for all paper figures. 600 dpi, Helvetica/Nimbus Sans fallback."""
import matplotlib as mpl

DPI = 600

# Standard categorical palette used across Fig 2–6
CATEGORY_COLORS = {
    'Cat1':     '#C0392B',   # red
    'Cat0':     '#2166AC',   # blue
    'Unknown':  '#1cb0b9',   # teal
}
WITHIN_COLOR  = '#d98b80'    # salmon
BETWEEN_COLOR = '#9aa0a8'    # grey

def apply_style():
    mpl.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'],
        'font.size':         11,
        'axes.labelweight':  'bold',
        'axes.titleweight':  'bold',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.8,
        'xtick.direction':   'out',
        'ytick.direction':   'out',
        'figure.dpi':        150,   # on-screen; final saves use DPI
        'savefig.dpi':       DPI,
        'pdf.fonttype':      42,    # embed TrueType (editable in Illustrator)
        'ps.fonttype':       42,
    })
