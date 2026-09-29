"""Manuscript figures (vector PDF).

   Palette: black, red and blue only.  Every colour is redundant with a dash pattern or a
   marker, so nothing is lost in greyscale printing.
     BLUE (#2a78d6) = water (free surface, pore-pressure profile, pressure field)
     RED  (#d62728) = the drawdown response (deformed shape, uniform mode, local thrust)
   Theory and FE are told apart by curve versus marker, never by colour.
   Grey (INK3) is reserved for guide lines only (grids, datums, reference levels).
   All text is black.
"""
from __future__ import annotations
import json, sys, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib import gridspec
import os as _os, pathlib as _pl
# Repository root: override with TD_ROOT if the tree is moved.
ROOT = _os.environ.get("TD_ROOT") or str(_pl.Path(__file__).resolve().parents[2])

OUT = Path(f"{ROOT}/paper/figs/out"); OUT.mkdir(parents=True, exist_ok=True)
BLUE, RED = "#2a78d6", "#d62728"            # black, red, blue is the whole palette

# One token per role.  Nothing in this file sets a size or a width literally.
FS_PANEL, FS_TEXT, FS_SMALL = 8.5, 7.5, 7.0      # panel label / annotation / secondary label
LW_DATA, LW_GEOM, LW_BOX, LW_GUIDE = 1.4, 1.4, 0.7, 0.6
PANEL_DX, PANEL_DY = 40, 2          # offset in points from the axes top-left corner
LEG   = dict(fontsize=FS_TEXT, handlelength=2.2, borderaxespad=0.5,
             frameon=True, facecolor="white", edgecolor="#0b0b0b", framealpha=1.0,
             fancybox=False)
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK, "axes.linewidth": LW_BOX,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "legend.frameon": False, "pdf.fonttype": 42, "figure.dpi": 150,
})
W1, W2 = 5.5, 5.5   # inches (single-column preprint ~ 140 mm)
from matplotlib.colors import LinearSegmentedColormap
WATER = LinearSegmentedColormap.from_list("water", ["#ffffff", "#2a78d6"])   # white to the paper blue

def panel(ax, s, dx=PANEL_DX):
    """Panel label at the top-left of the panel, level with the top of the axes
    and in the same column as the y-axis label."""
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction",
                xytext=(-dx, PANEL_DY), textcoords="offset points",
                fontsize=FS_PANEL, ha="left", va="bottom", annotation_clip=False)

def legend(ax, **kw):
    """Axes legend with the same box frame as the axes."""
    lg = ax.legend(**{**LEG, **kw})
    lg.get_frame().set_linewidth(LW_BOX)
    return lg


def spread_legend(fig, axs, h, l, ncol, y=0.0, height=0.055):
    """One legend under the panels, spanning from the left edge of the first axes to the
    right edge of the last."""
    x0 = axs[0].get_position().x0
    x1 = axs[-1].get_position().x1
    lg = fig.legend(h, l, loc="upper center", ncol=ncol, frameon=True, fancybox=False,
                    facecolor="white", edgecolor=INK, framealpha=1.0, fontsize=FS_TEXT,
                    handlelength=2.2, mode="expand", borderaxespad=0.0,
                    bbox_to_anchor=(x0, y, x1 - x0, height), bbox_transform=fig.transFigure)
    lg.get_frame().set_linewidth(LW_BOX)
    return lg


def clean(ax):
    """Full black box frame on all four sides."""
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True); ax.spines[s].set_color(INK); ax.spines[s].set_linewidth(LW_BOX)

# --------------------------------------------------------------------------- fig 1: configurations
def fig_configs():
    fig, axs = plt.subplots(1, 3, figsize=(W1, 2.9))
    titles = ["(I)", "(II)", "(III)"]
    for ax, ttl in zip(axs, titles):
        ax.set_aspect("equal"); ax.axis("off"); panel(ax, ttl, dx=0)
    # (I)
    ax = axs[0]
    ax.add_patch(Rectangle((0, 0), 4, 4, fc="none", ec=INK, lw=LW_BOX))
    ax.add_patch(Circle((2, 2), 0.55, fc="white", ec=INK, lw=LW_GEOM))
    for (x, y, dx, dy) in [(2, 4.35, 0, -0.3), (2, -0.35, 0, 0.3), (-0.35, 2, 0.3, 0), (4.35, 2, -0.3, 0)]:
        ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle="-|>", mutation_scale=7, color=INK, lw=LW_BOX))
    ax.text(2, -1.1, "$\\Delta p=-\\Delta u$ uniform\n$\\Delta\\sigma'=\\Delta u$ isotropic", ha="center", va="top", fontsize=FS_TEXT, color=INK)
    ax.set_xlim(-1.0, 5.0); ax.set_ylim(-2.4, 4.8)
    # (II)
    ax = axs[1]
    ax.add_patch(Rectangle((0, 0), 4, 4, fc="none", ec=INK, lw=LW_BOX))
    ax.add_patch(Circle((2, 2), 0.55, fc="white", ec=INK, lw=LW_GEOM))
    for x in (0, 4):
        for y in (0.7, 2, 3.3):
            ax.add_patch(Circle((x, y), 0.09, fc="white", ec=INK, lw=LW_BOX))
    for (x, y, dx, dy) in [(2, 4.35, 0, -0.3), (2, -0.35, 0, 0.3)]:
        ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle="-|>", mutation_scale=7, color=INK, lw=LW_BOX))
    ax.text(2, -1.1, "$\\Delta\\sigma'_v=\\Delta u$\n$\\Delta\\sigma'_h=K_0^{e}\\Delta u$", ha="center", va="top", fontsize=FS_TEXT, color=INK)
    ax.text(4.25, 2, r"$u_x=0$", fontsize=FS_SMALL, color=INK, va="center")
    ax.set_xlim(-1.0, 5.0); ax.set_ylim(-2.4, 4.8)
    # (III)
    ax = axs[2]
    ax.add_patch(Rectangle((0, 0), 4, 4, fc="none", ec=INK, lw=LW_BOX))
    ax.add_patch(Rectangle((0, -0.45), 4, 0.45, fc="none", ec=INK, lw=LW_BOX, hatch="...."))
    ax.plot([0, 4], [4, 4], color=BLUE, lw=LW_DATA)
    ax.add_patch(Circle((2.0, 2.5), 0.5, fc="white", ec=INK, lw=LW_GEOM))
    # linear drawdown profile on the right
    ax.plot([4.25, 4.25], [0, 4], color=INK, lw=LW_GUIDE)
    ax.plot([4.25, 5.05], [4, 0], color=BLUE, lw=LW_DATA)
    ax.text(4.60, 4.18, r"$\Delta p(z)$", fontsize=FS_SMALL, color=INK, ha="center", va="bottom")
    ax.plot([4.25, 5.05], [0, 0], color=INK, lw=LW_GUIDE)      # the scale of the profile
    ax.text(5.05, -0.18, r"$-\Delta u$", fontsize=FS_SMALL, color=INK, ha="center", va="top")
    ax.text(1.45, 4.18, r"free surface, $p=0$", fontsize=FS_SMALL, color=INK, ha="center", va="bottom")
    ax.text(2, -1.1, "aquifer, $p=-\\Delta u$\n$\\Delta p(z)$ linear, $\\Delta u_\\ell=\\Delta u\\,z_t/H$", fontsize=FS_TEXT, color=INK, ha="center", va="top")
    # layer thickness H and tunnel axis depth z_t (both measured from the free surface)
    # layer thickness, dimensioned outside the box
    for yy in (0, 4):
        ax.plot([-0.34, 0], [yy, yy], color=INK, lw=LW_GUIDE)
    ax.annotate("", xy=(-0.26, 0), xytext=(-0.26, 4),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=LW_GUIDE))
    ax.text(-0.26, 2.0, r"$H$", fontsize=FS_SMALL, color=INK, va="center", ha="center",
            bbox=dict(fc="white", ec="none", pad=0.8))
    ax.plot([1.45, 2.55], [2.5, 2.5], color=INK, lw=LW_GUIDE)
    ax.annotate("", xy=(2.0, 4), xytext=(2.0, 2.5), arrowprops=dict(arrowstyle="<->", color=INK, lw=LW_GUIDE))
    ax.text(2.14, 3.25, r"$z_t$", fontsize=FS_SMALL, color=INK, va="center")
    ax.set_xlim(-0.6, 5.4); ax.set_ylim(-2.4, 4.8)
    fig.savefig(OUT / "fig_configs.pdf", bbox_inches="tight"); plt.close(fig)

# --------------------------------------------------------------------------- fig 2: uniform mode (3b)
def fig_n0():
    R = json.load(open(f"{ROOT}/proj/drawdown_3b/dd_results.json"))
    nu = 0.34
    cs = np.logspace(-2, 2, 200)
    dq = (1 - 2 * nu) / (1 + cs); I0 = cs / (1 + cs)
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.70))
    for ax, y, key, lab in [(axs[0], dq, "dq_over_du", r"$\Delta q_0/\Delta u$"),
                            (axs[1], I0, "I0", r"$I_0 = u_r(R)/u_{\rm ff}(R)$")]:
        ax.semilogx(cs, y, "-", color=INK, lw=LW_DATA, label="closed form (I)")
        xs = [r["c_star"] for r in R]; ys = [r[key]["fe"] for r in R]
        ax.plot(xs, ys, "o", mfc="white", mec=INK, mew=1.1, ms=6.5, ls="none", zorder=5,
                clip_on=False, label="coupled FE (I)")
        ax.axvline(1.0, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)   # the transition
        ax.set_xlabel(r"$c^* = C(1-\nu')$"); ax.set_ylabel(lab); clean(ax)
    # condition (II): both closed forms are identically zero for any stiffness (Proposition 1)
    K0 = json.load(open(f"{ROOT}/proj/drawdown_k0/k0_results.json"))
    for ax, k in [(axs[0], "dq0_over_du_hm"), (axs[1], "u_r45_hm_over_uff")]:
        ax.semilogx([1e-2, 1e2], [0, 0], "-", color=RED, lw=LW_DATA, zorder=6,
                    label="closed form (II)")
        ax.plot([r["c_star"] for r in K0], [r["n0"][k] for r in K0], "s", mfc="white",
                mec=RED, mew=1.1, ms=5.5, ls="none", zorder=6, clip_on=False,
                label="coupled FE (II)")
    # the transition value itself, only where it has one
    axs[1].plot([1e-2, 1.0], [0.5, 0.5], color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
    axs[1].text(0.014, 0.53, r"$I_0=1/2$", fontsize=FS_TEXT, color=INK, va="bottom")
    panel(axs[0], "(a)")
    panel(axs[1], "(b)")
    axs[0].set_ylim(0, 0.35); axs[1].set_ylim(0, 1)
    for ax in axs: ax.set_xlim(1e-2, 1e2)
    h, l = axs[0].get_legend_handles_labels()
    order = [0, 2, 1, 3]                      # closed form (I), (II) then coupled FE (I), (II)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.075, 1, 1))
    spread_legend(fig, axs, [h[i] for i in order], [l[i] for i in order], ncol=4,
                  y=0.0, height=0.062)
    fig.savefig(OUT / "fig_n0.pdf", bbox_inches="tight"); plt.close(fig)

# --------------------------------------------------------------------------- fig 3: pressure field (3d)
def fig_pressure():
    import pyvista as pv
    files = sorted(glob.glob(f"{ROOT}/proj/drawdown_3d/runs/tun_step/out/dd3d_ts_*_t_*.vtu"),
                   key=lambda f: float(re.search(r"_t_([0-9.e+-]+)\.vtu", f).group(1)))
    m = pv.read(files[-1])
    ids = set()
    for ci in range(m.n_cells): ids.update(int(i) for i in m.get_cell(ci).point_ids[:4])
    ids = np.array(sorted(ids)); P = m.points[ids]; p = np.asarray(m.point_data["pressure"])[ids] / 1e5
    dom = pv.read(f"{ROOT}/proj/drawdown_3d/runs/tun_step/domain.vtu")
    mid = np.asarray(dom.cell_data["MaterialIDs"])
    # ground corner nodes only (exclude lining nodes, where p is inert)
    zt, Ro, Ri = 15.0, 5.125, 4.875
    rr = np.hypot(P[:, 0], P[:, 1] + zt)
    sel = rr > Ro - 1e-3
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(P[sel, 0], P[sel, 1])
    # mask triangles inside the tunnel
    cx = P[sel, 0][tri.triangles].mean(axis=1); cy = P[sel, 1][tri.triangles].mean(axis=1)
    tri.set_mask(np.hypot(cx, cy + zt) < Ro)
    # explicit axes rectangles so that (a) and (b) have the same height and the
    # z tick labels of the two panels line up
    FW, FH = W1, 4.15
    def rect(x, y, w, h): return [x / FW, y / FH, w / FW, h / FH]
    HAX = 2.95                       # axes height, inches
    WL = HAX * 30.0 / 40.0           # equal aspect over the 30 m by 40 m window
    X0, Y0 = 0.62, 1.00
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes(rect(X0, Y0, WL, HAX))
    cax = fig.add_axes(rect(X0, Y0 - 0.62, WL, 0.14))
    XR = X0 + WL + 0.95
    ax2 = fig.add_axes(rect(XR, Y0, FW - XR - 0.16, HAX))
    lv = np.linspace(0, 1, 11)      # one band per drawn isoline
    cf = ax.tricontourf(tri, -p[sel], levels=lv, cmap=WATER)      # drawdown fraction (positive)
    # filled contours leave pale hairline seams between the bands in vector output
    try: cf.set_edgecolor("face")
    except AttributeError:
        for c in cf.collections: c.set_edgecolor("face")
    ax.tricontour(tri, -p[sel], levels=np.linspace(0, 1, 11), colors=INK, linewidths=LW_GUIDE)
    ax.add_patch(Circle((0, -zt), Ri, fc="white", ec=INK, lw=LW_GEOM, zorder=5))
    ax.add_patch(Circle((0, -zt), Ro, fc="none", ec=INK, lw=LW_GEOM, zorder=5))
    ax.set_xlim(0, 30); ax.set_ylim(-40, 0)
    ax.set_aspect("equal", adjustable="box"); ax.grid(False)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); clean(ax)
    panel(ax, "(a)")
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label(r"drawdown fraction $-\Delta p/\Delta u$", fontsize=FS_TEXT)
    cb.ax.tick_params(labelsize=FS_SMALL)
    ax.text(0.6, -8.3, "crown 0.144", fontsize=FS_TEXT, color=INK, zorder=6, va="bottom",
            bbox=dict(fc="white", ec="none", pad=0.8))
    ax.text(0.6, -23.0, "invert 0.626", fontsize=FS_TEXT, color=INK, zorder=6, va="top",
            bbox=dict(fc="white", ec="none", pad=0.8))
    # profiles: through tunnel (x=0) vs far column
    for x0, lab, col, ls in [(108.0, "far field", INK, "--"), (0.0, "tunnel axis", INK, "-")]:
        s2 = np.abs(P[:, 0] - x0) < (1.5 if x0 > 1 else 0.3)
        zz, pp = P[s2, 1], -p[s2]
        if x0 == 0.0:
            s3 = rr[s2] > Ro - 1e-3; zz, pp = zz[s3], pp[s3]
        o = np.argsort(zz); zz, pp = zz[o], pp[o]
        if x0 == 0.0:
            up = zz > -zt; dn = zz < -zt
            ax2.plot(pp[up], zz[up], ls, color=col, lw=LW_DATA, label=lab); ax2.plot(pp[dn], zz[dn], ls, color=col, lw=LW_DATA)
        else:
            ax2.plot(pp, zz, ls, color=col, lw=LW_DATA, label=lab)
    ax2.axhspan(-zt - Ro, -zt + Ro, color="#f1f0ec", zorder=0)
    ax2.text(0.03, -zt, "tunnel", fontsize=FS_TEXT, color=INK, va="center")
    ax2.set_xlim(0, 1); ax2.set_ylim(-40, 0); ax2.set_yticks(ax.get_yticks())
    ax2.set_xlabel(r"$-\Delta p/\Delta u$"); ax2.set_ylabel("z (m)")
    panel(ax2, "(b)")
    legend(ax2, loc="upper right")
    clean(ax2); ax2.set_ylim(-40, 0)
    fig.savefig(OUT / "fig_pressure.pdf", bbox_inches="tight"); plt.close(fig)

# --------------------------------------------------------- fig: section forces around the ring
def fig_ring():
    """Finite layer: T and M around the ring, against the n=2 and the n=2+n=3 reconstruction."""
    sys.path.insert(0, f"{ROOT}/proj/drawdown_3d")
    import pyvista as pv, drawdown_3d as D
    c = D.Case3d(name="tun_step")
    fs = sorted(glob.glob(f"{ROOT}/proj/drawdown_3d/runs/tun_step/out/dd3d_ts_*_t_*.vtu"),
                key=lambda f: float(re.search(r"_t_([0-9.e+-]+)\.vtu", f).group(1)))
    th, T, M = D.lining_forces_half(c, pv.read(fs[-1]))
    A = np.column_stack([np.ones_like(th), np.sin(th), np.cos(2 * th),
                         np.sin(3 * th), np.cos(4 * th)])
    cT, *_ = np.linalg.lstsq(A, T, rcond=None)
    cM, *_ = np.linalg.lstsq(A, M, rcond=None)
    deg = np.degrees(th)
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.5))
    series = [(T / 1e3, cT, r"$T$ (kN/m)"), (M / 1e3, cM, r"$M$ (kNm/m)")]
    D2, D23 = (0, (5.5, 2.2)), (0, (4.0, 1.6, 1.0, 1.6))
    for ax, (y, cc, lab) in zip(axs, series):
        ax.plot(deg, y, "-", color=INK, lw=LW_DATA, label="coupled FE")
        ax.plot(deg, cc[2] * np.cos(2 * th) / 1e3, ls=D2, color=BLUE, lw=LW_DATA,
                label=r"$n=2$")
        ax.plot(deg, (cc[2] * np.cos(2 * th) + cc[3] * np.sin(3 * th)) / 1e3, ls=D23,
                color=RED, lw=LW_DATA, label=r"$n=2$ and $n=3$")
        ax.axhline(0, color=INK3, lw=LW_GUIDE, zorder=0)
        ax.set_xlim(-90, 90); ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_xticklabels(["invert", "", "springline", "", "crown"])
        ax.set_ylabel(lab); clean(ax)
    panel(axs[0], "(a)"); panel(axs[1], "(b)")
    axs[0].set_ylim(-80, 120); axs[1].set_ylim(-20, 30)
    axs[0].set_yticks([-80, -40, 0, 40, 80, 120]); axs[1].set_yticks([-20, -10, 0, 10, 20, 30])
    for ax in axs:
        lg = ax.legend(loc="upper right",
                       **{**LEG, "handlelength": 1.6, "handletextpad": 0.5,
                          "borderpad": 0.35, "labelspacing": 0.35, "borderaxespad": 0.4})
        lg.get_frame().set_linewidth(LW_BOX)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_ring.pdf", bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- fig 4: transient
def fig_transient():
    d3 = json.load(open(f"{ROOT}/proj/drawdown_3d/dd3d_results.json"))["tun_step"]["hist"]
    K0 = json.load(open(f"{ROOT}/proj/drawdown_k0/k0_results.json"))[0]
    k0 = K0["hist"]
    B3 = json.load(open(f"{ROOT}/proj/drawdown_3b/dd_results.json"))[0]
    b3 = B3["hist"]
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.6))

    # (a) conditions (I) and (II)
    t2 = np.array([h["t"] for h in k0]) / K0["T_ref"]
    T0 = np.array([h["T0"] for h in k0]); T2v = np.array([h["T2"] for h in k0])
    tb = np.array([h["t"] for h in b3]) / B3["T_ref"]; qb = np.array([h["T0"] for h in b3])
    axs[0].semilogx(t2[1:], T2v[1:] / T2v[-1], "-", color=INK, lw=LW_DATA, label=r"(II) $T_2$")
    axs[0].semilogx(t2[1:], T0[1:] / T2v[-1], "--", color=RED, lw=LW_DATA,
                    label=r"(II) $T_0/T_2(\infty)$")
    axs[0].semilogx(tb[1:], qb[1:] / qb[-1], "-.", color=BLUE, lw=LW_DATA, label=r"(I) $\Delta q_0$")
    axs[0].axhline(0, color=INK3, lw=LW_GUIDE)
    panel(axs[0], "(a)")
    legend(axs[0], loc="upper left")

    # (b) condition (III): mode resultants in black, local thrusts in colour
    t = np.array([h["t_over_Tref"] for h in d3])
    for key, lab, col, ls in [("T2", r"$T_2$", INK, "-"), ("M2", r"$M_2$", INK, "--"),
                              ("T_crn", r"$T$ crown", RED, "-."), ("T_inv", r"$T$ invert", BLUE, ":")]:
        v = np.array([h[key] for h in d3])
        axs[1].semilogx(t[1:], v[1:] / v[-1], ls, color=col, lw=LW_DATA, label=lab)
    panel(axs[1], "(b)")
    legend(axs[1], loc="upper left")

    for ax in axs:
        clean(ax); ax.set_ylim(-0.08, 1.0)
        ax.set_xlabel(r"$t/t_{\rm ref}$"); ax.set_xlim(1e-4, 1.0)
        ax.set_ylabel(r"$X\,/\,X(\infty)$")
    fig.tight_layout(w_pad=2.2); fig.savefig(OUT / "fig_transient.pdf", bbox_inches="tight"); plt.close(fig)

# --------------------------------------------------------------------------- fig 5: design chart (exact solution)
def fig_chart(recompute=True):
    sys.path.insert(0, f"{ROOT}/paper/theory"); import exact_ring as ex
    cache = Path(f"{ROOT}/paper/theory/chart_data.json")
    E, nul, R, t = 48e6, 0.15, 5.0, 0.25
    nus = [0.25, 0.30, 0.34, 0.40, 0.45]
    Fs = np.logspace(0.0, 4.0, 17)
    if recompute or not cache.exists():
        data = {}
        for nu in nus:
            K0p = nu / (1 - nu); rows = []
            for F in Fs:
                El = E * R**3 * (1 - nul**2) / (F * t**3 / 12 * (1 - nu**2))
                s = ex.solve_external(E, nu, El, nul, R - t / 2, R + t / 2, -(1 + K0p) / 2 * 1.0, (1 - K0p) / 2 * 1.0)  # Δu_l = 1
                rows.append(dict(F=F, T2n=s["T2"] / R, M2n=s["M2"] / R**2, I2=s["I2"]))
            data[str(nu)] = rows
        cache.write_text(json.dumps(data))
    data = json.load(open(cache))
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.6))
    # the dash gets shorter as nu\' grows
    styles = ["-", (0, (6, 1.6)), (0, (3.6, 1.6)), (0, (2.2, 1.6)), (0, (0.9, 1.5))]
    for (nu, rows), st in zip(data.items(), styles):
        F = [r["F"] for r in rows]
        axs[0].semilogx(F, [r["T2n"] for r in rows], ls=st, color=INK, lw=LW_DATA)
        axs[1].loglog(F, [r["M2n"] for r in rows], ls=st, color=INK, lw=LW_DATA)
        axs[1].lines[-1].set_label(rf"$\nu'={nu}$")
    # FE points: 3c (t=0.125) and 3d (t=0.25)
    k0 = json.load(open(f"{ROOT}/proj/drawdown_k0/k0_results.json"))
    lab2 = "coupled FE (II)"
    for r_ in k0:
        F_ = r_["F"]
        if F_ > 1e4:            # outside the chart range: F alone no longer fixes T2 there
            continue
        T2n = r_["n2"]["T2_hm"] / (1e5 * 5.0); M2n = r_["n2"]["M2_hm"] / (1e5 * 25.0)
        axs[0].plot(F_, T2n, "s", mfc="white", mec=INK, mew=1.1, ms=6, ls="none", zorder=5,
                    clip_on=False, label=lab2)
        axs[1].plot(F_, M2n, "s", mfc="white", mec=INK, mew=1.1, ms=6, ls="none", zorder=5,
                    clip_on=False)
        lab2 = None
    # condition (III) coupled drawdown runs: the reference case and the sweep cases that
    # change F or nu' (the z_t and R cases repeat the reference point on this chart)
    d3 = json.load(open(f"{ROOT}/proj/drawdown_3d/dd3d_results.json"))["tun_step"]
    pts = [(d3["params"]["F"], d3["params"]["du_local"], 5.0, d3["hist"][-1])]
    sw = json.load(open(f"{ROOT}/proj/drawdown_3d/n3_sweep_b.json"))
    for k in ("thick", "nu25", "nu45"):
        v = sw[k]
        pts.append((v["params"]["F"], v["params"]["du_local"], 5.0, v["hist"][-1]))
    lf = json.load(open(f"{ROOT}/proj/drawdown_3d/lowF_result.json"))["softF4"]
    pts.append((lf["params"]["F"], lf["params"]["du_local"], 5.0, lf["hist"][-1]))
    lab3 = "coupled FE (III)"
    for F_, dul, R_, e in pts:
        axs[0].plot(F_, e["T2"] / (dul * R_), "o", mfc="white", mec=INK, mew=1.1,
                    ms=6.5, ls="none", zorder=5, clip_on=False, label=lab3)
        axs[1].plot(F_, e["M2"] / (dul * R_**2), "o", mfc="white", mec=INK, mew=1.1,
                    ms=6.5, ls="none", zorder=5, clip_on=False)
        lab3 = None
    axs[0].set_ylabel(r"$T_2/(\Delta u_\ell R)$"); axs[1].set_ylabel(r"$M_2/(\Delta u_\ell R^2)$")
    panel(axs[0], "(a)")
    panel(axs[1], "(b)")
    for ax in axs: ax.set_xlabel(r"flexibility ratio $F$"); clean(ax); ax.set_xlim(1, 1e4)
    for ax in axs: ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4])
    axs[0].set_ylim(0, 0.55); axs[1].set_ylim(1e-4, 1.0)
    hl, ll = axs[1].get_legend_handles_labels()          # the five nu' curves
    hm, lm = axs[0].get_legend_handles_labels()          # the two marker series
    # two rows, filled column by column: nu' curves on the first row, markers on the second
    h = [hl[0], hl[4], hl[1], hm[0], hl[2], hm[1], hl[3]]
    l = [ll[0], ll[4], ll[1], lm[0], ll[2], lm[1], ll[3]]
    fig.tight_layout(w_pad=2.0, rect=(0, 0.13, 1, 1))
    # the legend spans exactly from the left edge of the (a) axes to the right edge of (b)
    x0 = axs[0].get_position().x0
    x1 = axs[1].get_position().x1
    leg = fig.legend(h, l, loc="upper center", ncol=4, frameon=True, fancybox=False,
                     facecolor="white", edgecolor=INK, framealpha=1.0, fontsize=FS_TEXT,
                     handlelength=2.2, mode="expand", borderaxespad=0.0,
                     bbox_to_anchor=(x0, 0.0, x1 - x0, 0.115), bbox_transform=fig.transFigure)
    leg.get_frame().set_linewidth(LW_BOX)
    fig.savefig(OUT / "fig_chart.pdf", bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------- fig: theory map
def fig_theory():
    """Mechanics of \S3, drawn in the idiom of fig_configs."""
    from matplotlib.patches import Ellipse
    fig, axs = plt.subplots(1, 3, figsize=(W1, 3.15))
    titles = ["(a)", "(b)", "(c)"]
    for ax, ttl in zip(axs, titles):
        ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)
        panel(ax, ttl, dx=0)
        ax.set_xlim(-1.0, 5.0); ax.set_ylim(-4.9, 4.8)
        ax.add_patch(Rectangle((0, 0), 4, 4, fc="none", ec=INK, lw=LW_BOX))
        ax.add_patch(Circle((2, 2), 0.55, fc="white", ec=INK, lw=LW_GEOM))

    def arrows(ax, spec, lw=LW_BOX):
        for (x, y, dx, dy) in spec:
            ax.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle="-|>",
                         mutation_scale=7, color=INK, lw=lw))
    VERT = [(2, 4.35, 0, -0.3), (2, -0.35, 0, 0.3)]
    HORZ = [(-0.35, 2, 0.3, 0), (4.35, 2, -0.3, 0)]

    def rollers(ax):
        for x in (0, 4):
            for y in (0.7, 2, 3.3):
                ax.add_patch(Circle((x, y), 0.09, fc="white", ec=INK, lw=LW_BOX))

    def bars(ax, skel, water):
        x0, y0 = 0.45, -2.85
        ax.plot([x0, x0], [y0 - 0.62, y0 + 0.62], color=INK3, lw=LW_GUIDE)
        for (val, dy, col, lab) in ((skel, 0.30, INK, "skeleton"),
                                    (water, -0.30, INK, "water")):
            ax.add_patch(FancyArrowPatch((x0, y0 + dy), (x0 + val, y0 + dy),
                         arrowstyle="-|>", mutation_scale=6, color=col, lw=LW_GEOM))
            ax.text(x0 + val + 0.12, y0 + dy, lab, fontsize=FS_SMALL, color=col,
                    va="center", ha="left")

    # ================================================================== (a)
    ax = axs[0]
    arrows(ax, VERT + HORZ)
    ax.add_patch(Circle((2, 2), 0.42, fc="none", ec=RED, lw=LW_GEOM, ls=(0, (2.6, 1.8))))
    ax.text(2, -0.9, "$\\Delta\\sigma\'=\\Delta u$ isotropic\n$p_m=\\Delta u$",
            ha="center", va="top", fontsize=FS_TEXT, color=INK)
    bars(ax, 2.30, 1.74)

    # ================================================================== (b)
    ax = axs[1]
    arrows(ax, VERT); rollers(ax)
    # the deformed shape coincides with the undeformed ring
    ax.add_patch(Circle((2, 2), 0.55, fc="none", ec=RED, lw=LW_GEOM, ls=(0, (2.4, 2.4)), zorder=6))
    ax.text(4.25, 2, r"$u_x=0$", fontsize=FS_SMALL, color=INK, va="center")
    ax.text(2, -0.9, "$\\Delta\\sigma\'_h=K_0^{e}\\Delta u$\n$p_m=\\Delta u/[2(1-\\nu\')]$",
            ha="center", va="top", fontsize=FS_TEXT, color=INK)
    bars(ax, 2.02, 2.02)
    ax.text(2, -4.15, "Proposition 1", ha="center", va="top", fontsize=FS_TEXT, color=INK)

    # ================================================================== (c)
    ax = axs[2]
    arrows(ax, VERT); arrows(ax, [(-0.35, 2, 0.16, 0), (4.35, 2, -0.16, 0)], lw=LW_GUIDE)
    ax.add_patch(Ellipse((2, 2), 2 * 0.55 * 1.28, 2 * 0.55 * 0.72, fc="none", ec=RED,
                         lw=LW_GEOM, ls=(0, (2.6, 1.8))))
    ax.text(2, -0.9, "$\\Delta\\sigma\'_v=\\Delta u$,  $\\Delta\\sigma\'_h=K_0^{e}\\Delta u$\n"
                     "$s=\\frac{1}{2}(1-K_0^{e})\\Delta u$",
            ha="center", va="top", fontsize=FS_TEXT, color=INK)
    ax.text(2, -2.75, "= external loading\nwith $P=\\Delta u_\\ell$",
            ha="center", va="top", fontsize=FS_TEXT, color=INK)
    ax.text(2, -4.15, "Proposition 2", ha="center", va="top", fontsize=FS_TEXT, color=INK)

    fig.savefig(OUT / "fig_theory.pdf", bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    which = sys.argv[1:] or ["configs", "n0", "pressure", "transient", "chart", "theory"]
    for w in which:
        globals()[f"fig_{w}"]() if w != "chart" else fig_chart(recompute=False)
        print("made", w, flush=True)
