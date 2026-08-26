#!/usr/bin/env python3
"""
Supplementary Figure — Heterogeneity of ssGSEA NES profiles across the FULL
pre-selection GO BP term set (2,692 terms x 40 chemicals).

Three panels:
  (1) Violin plot of pairwise NES-profile correlations for the 35 labeled
      chemicals, split into within-category vs between-category pairs
      (Mann-Whitney U test).
  (2) Compound x compound NES-profile correlation heatmap, ordered
      Category1 | Category0 | Unknown.
  (3) Histogram of the per-term cross-compound variance of NES, with the
      medians of the keyword-selected and ontology-selected developmental
      term subsets overlaid, to show developmental terms are not more
      variable than the genome-wide term pool.

------------------------------------------------------------------------------
INPUTS
  --nes   ssGSEA NES matrix, CSV with terms in rows and the 40 chemicals in
          columns, in the fixed order:
              Category1 (20 chemicals) | Category0 (15) | Unknown (5).
          Term names must carry their GO id in the form "... (GO:0001234)".
          (This is the same NES matrix used to build Fig. 2b; see NOTE below.)
  --keywords  SX_keyword_list.csv with columns [role, keyword_or_root] where
          role is one of: include / exclude / ontology_root.
  --obo   go-basic.obo (Gene Ontology). Download once, e.g.:
              curl -L -o go-basic.obo https://current.geneontology.org/ontology/go-basic.obo
  --out   output image path (default: SX_heterogeneity.png)

NOTE on the NES matrix
  The NES matrix is produced exactly as for Fig. 2b: per-gene Delta-rlog
  (compound minus the mean of the 6 DMSO controls) -> gseapy.ssgsea against
  GO_Biological_Process (sample_norm_method='rank') -> pivot to a
  (terms x chemicals) table. Absolute correlation values in panels 1-2 depend
  on the ssGSEA implementation/version; term selection, the pair counts, the
  keyword/ontology counts and the variance distribution are robust to it.

DEPENDENCIES
  python>=3.10, pandas, numpy, scipy, matplotlib
------------------------------------------------------------------------------
"""
import argparse
import itertools
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- fixed chemical grouping (column order of the NES matrix) ----
N_CAT1, N_CAT0, N_UNK = 20, 15, 5
CATEGORY = ["Cat1"] * N_CAT1 + ["Cat0"] * N_CAT0 + ["Unknown"] * N_UNK

VIOLIN_COLORS = ["#d98b80", "#9aa0a8"]      # within, between
ALL_COLOR, ONTO_COLOR, KW_COLOR = "#b8bcc4", "#c0392b", "#2f6fb0"


def go_id(term: str):
    """Extract the GO id embedded in a term name, e.g. 'Foo (GO:0001234)'."""
    m = re.search(r"\((GO:\d+)\)", term)
    return m.group(1) if m else None


def keyword_mask(terms, kw_csv):
    """Boolean mask of terms matching the include roots and not the exclude roots."""
    kw = pd.read_csv(kw_csv)
    inc = [k.lower() for r, k in zip(kw.role, kw.keyword_or_root) if r == "include"]
    exc = [k.lower() for r, k in zip(kw.role, kw.keyword_or_root) if r == "exclude"]

    def matched(term):
        t = re.sub(r"\s*\(GO:\d+\)", "", term).lower()
        if any(e in t for e in exc):
            return False
        return any(i in t for i in inc)

    return np.array([matched(t) for t in terms])


def ontology_descendants(obo_path, root="GO:0032502"):
    """All is_a + part_of descendants of `root` from a go-basic.obo file."""
    parents, cur = {}, None
    for line in open(obo_path, encoding="utf-8"):
        line = line.rstrip()
        if line == "[Term]":
            cur = {"id": None, "rel": [], "obs": False}
        elif cur is not None and line.startswith("id: GO:") and cur["id"] is None:
            cur["id"] = line[4:]
        elif cur is not None and line.startswith("is_a: GO:"):
            cur["rel"].append(line.split()[1])
        elif cur is not None and line.startswith("relationship: part_of GO:"):
            cur["rel"].append(line.split()[2])
        elif cur is not None and line.startswith("is_obsolete: true"):
            cur["obs"] = True
        elif line == "" and cur is not None and cur["id"]:
            if not cur["obs"]:
                parents[cur["id"]] = cur["rel"]
            cur = None
    children = defaultdict(list)
    for c, ps in parents.items():
        for p in ps:
            children[p].append(c)
    desc, dq = set(), deque([root])
    while dq:
        n = dq.popleft()
        for c in children.get(n, []):
            if c not in desc:
                desc.add(c)
                dq.append(c)
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nes", required=True, help="ssGSEA NES matrix CSV (terms x 40 chemicals)")
    ap.add_argument("--keywords", required=True, help="SX_keyword_list.csv")
    ap.add_argument("--obo", required=True, help="go-basic.obo")
    ap.add_argument("--root", default="GO:0032502", help="ontology root (developmental process)")
    ap.add_argument("--out", default="SX_heterogeneity.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    nes = pd.read_csv(args.nes, index_col=0)
    assert nes.shape[1] == N_CAT1 + N_CAT0 + N_UNK, "expected 40 chemical columns"
    X = nes.values                                   # (n_terms, 40)

    # ----- developmental term subsets -----
    kw_mask = keyword_mask(nes.index, args.keywords)
    desc = ontology_descendants(args.obo, args.root)
    onto_mask = np.array([go_id(t) in desc for t in nes.index])
    print(f"terms: {nes.shape[0]} | keyword-selected: {kw_mask.sum()} | "
          f"ontology-selected: {onto_mask.sum()}")

    # ----- panel 1: within vs between category pairwise correlation (35 labeled) -----
    lab = [i for i, c in enumerate(CATEGORY) if c != "Unknown"]
    lab_cat = [CATEGORY[i] for i in lab]
    corr35 = np.corrcoef(X[:, lab].T)
    within, between = [], []
    for a, b in itertools.combinations(range(len(lab)), 2):
        (within if lab_cat[a] == lab_cat[b] else between).append(corr35[a, b])
    within, between = np.array(within), np.array(between)
    dmean = within.mean() - between.mean()
    _, p = mannwhitneyu(within, between, alternative="two-sided")
    print(f"within n={len(within)} med={np.median(within):.3f} | "
          f"between n={len(between)} med={np.median(between):.3f} | "
          f"dmean={dmean:+.3f} MWU p={p:.3f}")

    # ----- panel 2: 40x40 compound correlation, already in category order -----
    corr40 = np.corrcoef(X.T)

    # ----- panel 3: per-term cross-compound variance -----
    var = X.var(axis=1)
    med_all = np.median(var)
    med_kw = np.median(var[kw_mask])
    med_on = np.median(var[onto_mask])
    print(f"variance median: all={med_all:.5f} ontology={med_on:.5f} keyword={med_kw:.5f}")

    # =========================== figure ===========================
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig = plt.figure(figsize=(19, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.15], wspace=0.28)
    fig.suptitle("Heterogeneity quantified on the FULL pre-selection term set "
                 "(2,692 GO BP terms)", fontsize=13, y=0.99)

    ax1 = fig.add_subplot(gs[0, 0])
    parts = ax1.violinplot([within, between], positions=[0, 1],
                           showmeans=False, showextrema=False, widths=0.85)
    for pc, c in zip(parts["bodies"], VIOLIN_COLORS):
        pc.set_facecolor(c); pc.set_edgecolor("none"); pc.set_alpha(0.9)
    for i, d in enumerate([within, between]):
        ax1.plot([i - 0.32, i + 0.32], [d.mean()] * 2, color="black", lw=2)
    ax1.axhline(0, ls="--", color="#888", lw=1)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([f"Within-category\n(n={len(within)})",
                         f"Between-category\n(n={len(between)})"])
    ax1.set_ylabel("Pairwise NES-profile correlation r")
    ax1.set_title(f"Category explains little of the profile\n"
                  f"(Δmean=+{dmean:.3f}, MWU p={p:.3f})", fontsize=10.5)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(corr40, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="equal")
    for b in (N_CAT1, N_CAT1 + N_CAT0):
        ax2.axhline(b - 0.5, color="black", lw=1)
        ax2.axvline(b - 0.5, color="black", lw=1)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Compound×compound NES correlation\n"
                  "(grouped: Cat1 | Cat0 | Unknown)", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("r")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(var, bins=np.linspace(0, 0.05, 45), color=ALL_COLOR, edgecolor="none")
    ax3.axvline(med_all, ls="--", color="#888", lw=1.2)
    ax3.axvline(med_on, color=ONTO_COLOR, lw=1.6)
    ax3.axvline(med_kw, color=KW_COLOR, lw=1.6)
    ax3.legend(handles=[
        Patch(facecolor=ALL_COLOR, label="All terms (n=%d)" % nes.shape[0]),
        plt.Line2D([], [], color=ONTO_COLOR, label="Ontology median (n=%d)" % onto_mask.sum()),
        plt.Line2D([], [], color=KW_COLOR, label="Keyword median (n=%d)" % kw_mask.sum()),
    ], fontsize=8.5, frameon=False, loc="upper right")
    ax3.set_xlim(0, 0.05)
    ax3.set_xlabel("Cross-compound variance of NES (per term)")
    ax3.set_ylabel("Density")
    ax3.set_title("Developmental terms are NOT more variable\n"
                  "than the genome-wide term pool", fontsize=10.5)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print("saved:", args.out)


if __name__ == "__main__":
    main()