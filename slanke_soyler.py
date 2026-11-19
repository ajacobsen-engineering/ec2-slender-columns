"""Kontroll av slanke betongsøyler etter NS-EN 1992-1-1, generell metode (5.8.6).

Søylen deles i snitt langs lengden. I hvert snitt hentes krumningen fra
ikke-lineær tverrsnittsanalyse (structuralcodes) og integreres til utbøyning.
2. ordens momentet oppdateres til det konvergerer. Kryp legges inn ved å
skalere betongtøyningene med (1+phi_ef); kryptallet kan angis manuelt eller
regnes etter Tillegg B.

Tar rektangulært og sirkulært tverrsnitt, leddlagret og utkrager, enakset og
toakset bøying, og sammenligner mot de forenklede metodene 5.8.7 og 5.8.8.

Enheter: N, mm, MPa (moment Nmm). Trykk er negativt i kall til structuralcodes.
Krever: structuralcodes, numpy, matplotlib.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

warnings.filterwarnings("ignore")

from structuralcodes.geometry import (
    CircularGeometry, RectangularGeometry,
    add_reinforcement_circle, add_reinforcement_line,
)
from structuralcodes.materials.concrete import create_concrete
from structuralcodes.materials.reinforcement import create_reinforcement
from structuralcodes.sections import BeamSection


@dataclass
class Soyle:
    # tverrsnitt
    section_type: str = "rect"       # 'rect' eller 'circle'
    b: float = 400.0                 # bredde [mm]
    h: float = 400.0                 # høyde [mm], bøying om y styres av h
    D: float = 400.0                 # diameter [mm] (circle)
    cover_c: float = 50.0            # kant til armeringssenter [mm]

    # lengde og opplagring
    l0: float = 6000.0               # effektiv lengde [mm]
    boundary: str = "pinned"         # 'pinned' eller 'cantilever'

    # materialer
    fck: float = 35.0
    fyk: float = 500.0
    Es: float = 200000.0
    ftk: float = 540.0
    epsuk: float = 0.0675

    # armering
    n_bars_per_side: int = 4         # stenger på hver bøyeside (rect)
    n_bars_side: int = 0             # ekstra stenger på sideflatene (rect)
    bar_dia: float = 25.0
    n_bars_circle: int = 8

    # laster (trykk angis positivt)
    N_Ed: float = 2500e3
    M0y_top: float = 80e6
    M0y_bot: float = 40e6
    M0z_top: float = 0.0
    M0z_bot: float = 0.0
    qy: float = 0.0                  # jevnt fordelt tverrlast [N/mm]
    Qy: float = 0.0                  # punktlast midt/tipp [N]
    qz: float = 0.0
    Qz: float = 0.0

    # kryp
    phi_ef: float = 1.0
    auto_creep: bool = False
    RH: float = 50.0
    t0: float = 28.0
    t_creep: float = 1e5
    cement_class: str = "N"
    M0_qp_ratio: float = 0.7         # M0Eqp/M0Ed til phi_ef (5.8.4)

    # beregning
    n_stations: int = 41
    tol_M2: float = 0.01
    max_iter: int = 100
    relax: float = 0.8


# --- kryptall etter Tillegg B --------------------------------------
def kryptall(fck, RH, h0, t0, t, cement="N"):
    fcm = fck + 8.0
    a_cem = {"S": -1.0, "N": 0.0, "R": 1.0}[cement]
    t0 = max(0.5, t0 * (9.0 / (2.0 + t0 ** 1.2) + 1.0) ** a_cem)

    if fcm <= 35.0:
        phi_RH = 1.0 + (1.0 - RH / 100.0) / (0.1 * h0 ** (1 / 3))
        bH = 1.5 * (1 + (0.012 * RH) ** 18) * h0 + 250.0
        bH = min(bH, 1500.0)
    else:
        a1, a2, a3 = [(35.0 / fcm) ** p for p in (0.7, 0.2, 0.5)]
        phi_RH = (1.0 + (1.0 - RH / 100.0) / (0.1 * h0 ** (1 / 3)) * a1) * a2
        bH = min(1.5 * (1 + (0.012 * RH) ** 18) * h0 + 250.0 * a3, 1500.0 * a3)

    phi0 = phi_RH * (16.8 / math.sqrt(fcm)) * (1.0 / (0.1 + t0 ** 0.2))
    dt = max(t - t0, 0.0)
    return phi0 * (dt / (bH + dt)) ** 0.3 if dt > 0 else 0.0


def phi_ef(s: Soyle) -> float:
    if not s.auto_creep:
        return s.phi_ef
    if s.section_type == "circle":
        h0 = s.D / 2.0
    else:
        h0 = 2.0 * (s.b * s.h) / (2.0 * (s.b + s.h))
    return kryptall(s.fck, s.RH, h0, s.t0, s.t_creep, s.cement_class) * s.M0_qp_ratio


# --- tverrsnitt ----------------------------------------------------
def bygg_tverrsnitt(s: Soyle, phi: float) -> BeamSection:
    f = 1.0 + phi
    conc = create_concrete(fck=s.fck, design_code="ec2_2004",
                           eps_c2=-0.002 * f, eps_cu2=-0.0035 * f)
    steel = create_reinforcement(fyk=s.fyk, Es=s.Es, ftk=s.ftk, epsuk=s.epsuk,
                                 design_code="ec2_2004")
    if s.section_type == "circle":
        geo = CircularGeometry(diameter=s.D, material=conc, concrete=True)
        geo = add_reinforcement_circle(geo, (0, 0), s.D / 2 - s.cover_c,
                                       s.bar_dia, steel, n=s.n_bars_circle)
    else:
        geo = RectangularGeometry(width=s.b, height=s.h, material=conc, concrete=True)
        c = s.cover_c
        xl, xr = -s.b / 2 + c, s.b / 2 - c
        yb, yt = -s.h / 2 + c, s.h / 2 - c
        geo = add_reinforcement_line(geo, (xl, yb), (xr, yb), s.bar_dia, steel, n=s.n_bars_per_side)
        geo = add_reinforcement_line(geo, (xl, yt), (xr, yt), s.bar_dia, steel, n=s.n_bars_per_side)
        if s.n_bars_side > 0:
            geo = add_reinforcement_line(geo, (xl, yb), (xl, yt), s.bar_dia, steel,
                                         n=s.n_bars_side, first=False, last=False)
            geo = add_reinforcement_line(geo, (xr, yb), (xr, yt), s.bar_dia, steel,
                                         n=s.n_bars_side, first=False, last=False)
    return BeamSection(geo)


def krumning(sec, N, My, Mz=0.0):
    """Krumning (chi_y, chi_z) og konvergensflagg for gitt N og signert moment."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sec.section_calculator.calculate_strain_profile(n=N, my=My, mz=Mz)
        return r.chi_y, r.chi_z, bool(r.converged)
    except Exception:
        return 0.0, 0.0, False


def M_Rd(sec, N, theta=0.0):
    """Momentkapasitet; 0 hvis aksiallasten alene sprenger tverrsnittet."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = sec.section_calculator.calculate_bending_strength(n=N, theta=theta)
        return math.hypot(u.m_y, u.m_z)
    except Exception:
        return 0.0


# --- geometri, slankhet, imperfeksjon ------------------------------
def tverrsnitt_data(s: Soyle):
    if s.section_type == "circle":
        Ac = math.pi * s.D ** 2 / 4
        i = s.D / 4
        As = s.n_bars_circle * math.pi * (s.bar_dia / 2) ** 2
        d = s.D / 2 + (s.D / 2 - s.cover_c) / 2
    else:
        Ac = s.b * s.h
        i = s.h / math.sqrt(12)
        As = (2 * s.n_bars_per_side + 2 * s.n_bars_side) * math.pi * (s.bar_dia / 2) ** 2
        d = s.h - s.cover_c
    return Ac, i, As, d


def e_imperf(s: Soyle) -> float:
    l_m = s.l0 / 1000
    alpha_h = min(1.0, max(2 / 3, 2 / math.sqrt(l_m)))
    return (1 / 200) * alpha_h * s.l0 / 2


def slankhet(s: Soyle, phi: float):
    Ac, i, As, d = tverrsnitt_data(s)
    lam = s.l0 / i
    fcd = create_concrete(fck=s.fck, design_code="ec2_2004").fcd()
    n = abs(s.N_Ed) / (Ac * fcd)
    A = 1 / (1 + 0.2 * phi)
    rm = s.M0y_bot / s.M0y_top if abs(s.M0y_top) > 1e-9 else 0.0
    C = max(0.7, min(1.7 - rm, 1.7))
    lam_lim = 20 * A * 1.1 * C / math.sqrt(n) if n > 0 else 1e9
    return lam, lam_lim, n, fcd


# --- momentfordeling og integrasjon --------------------------------
def M0_fordeling(x, L, M_top, M_bot, q, Q, boundary):
    if boundary == "cantilever":
        return M_top + Q * (L - x) + q * (L - x) ** 2 / 2
    M = M_bot + (M_top - M_bot) * (x / L)
    M = M + q * x * (L - x) / 2
    return M + Q * np.where(x <= L / 2, x / 2, (L - x) / 2)


def integrer(kappa, dx, boundary):
    """Utbøyning fra krumning, med samme fortegn som kappa (forsterkende)."""
    n = len(kappa)
    if n < 3:
        return np.zeros(n)
    if boundary == "cantilever":
        vp = np.concatenate([[0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * dx)])
        return np.concatenate([[0.0], np.cumsum(0.5 * (vp[1:] + vp[:-1]) * dx)])
    # leddlagret: v'' = -kappa, v=0 i endene
    m = n - 2
    A = np.diag(np.full(m, -2.0)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)
    v = np.zeros(n)
    v[1:-1] = np.linalg.solve(A, -kappa[1:-1] * dx * dx)
    return v


def arm(y, boundary):
    return y[-1] - y if boundary == "cantilever" else y


@dataclass
class Resultat:
    stabil: bool
    bruddtype: str
    aarsak: str
    iter: int
    phi_ef: float
    e_i: float
    e2: float
    M_Ed: float
    x_krit: float
    M_Rd: float
    utn: float
    x: np.ndarray = field(repr=False, default=None)
    My: np.ndarray = field(repr=False, default=None)
    Mz: np.ndarray = field(repr=False, default=None)
    vy: np.ndarray = field(repr=False, default=None)
    vz: np.ndarray = field(repr=False, default=None)
    ky: np.ndarray = field(repr=False, default=None)
    kz: np.ndarray = field(repr=False, default=None)
    M2_hist: list = field(repr=False, default=None)


# --- generell metode (5.8.6 / 5.8.9) -------------------------------
def generell_metode(s: Soyle, verbose=False) -> Resultat:
    phi = phi_ef(s)
    sec = bygg_tverrsnitt(s, phi)
    N = abs(s.N_Ed)

    L = s.l0 if s.boundary == "pinned" else s.l0 / 2
    x = np.linspace(0, L, s.n_stations)
    dx = x[1] - x[0]

    M0y = M0_fordeling(x, L, s.M0y_top, s.M0y_bot, s.qy, s.Qy, s.boundary)
    M0z = M0_fordeling(x, L, s.M0z_top, s.M0z_bot, s.qz, s.Qz, s.boundary)

    # imperfeksjon i den mest ugunstige retningen (5.8.9(2))
    ei = e_imperf(s)
    shape = (x / L) if s.boundary == "cantilever" else np.sin(math.pi * x / L)
    ei_y, ei_z = np.zeros_like(x), np.zeros_like(x)
    if max(abs(s.M0y_top), abs(s.M0y_bot)) >= max(abs(s.M0z_top), abs(s.M0z_bot)):
        m = s.M0y_top if abs(s.M0y_top) >= abs(s.M0y_bot) else s.M0y_bot
        ei_y = math.copysign(1, m or 1) * ei * shape
    else:
        m = s.M0z_top if abs(s.M0z_top) >= abs(s.M0z_bot) else s.M0z_bot
        ei_z = math.copysign(1, m or 1) * ei * shape

    vy, vz = np.zeros_like(x), np.zeros_like(x)
    ky, kz = np.zeros_like(x), np.zeros_like(x)
    hist = []

    def pakk(stabil, btype, aarsak, it):
        My = M0y + N * arm(ei_y + vy, s.boundary)
        Mz = M0z + N * arm(ei_z + vz, s.boundary)
        Mres = np.hypot(My, Mz)
        ic = int(np.argmax(Mres))
        theta = math.atan2(Mz[ic], My[ic]) if Mres[ic] > 1e-9 else 0.0
        cap = M_Rd(sec, -N, theta)
        e2 = float(np.max(np.hypot(vy, vz)))
        return Resultat(stabil, btype, aarsak, it, phi, ei, e2, float(Mres[ic]),
                        float(x[ic]), cap, Mres[ic] / cap if cap > 0 else float("inf"),
                        x, My, Mz, vy, vz, ky, kz, hist)

    M2_prev = None
    for it in range(1, s.max_iter + 1):
        My = M0y + N * arm(ei_y + vy, s.boundary)
        Mz = M0z + N * arm(ei_z + vz, s.boundary)

        for j in range(len(x)):
            ky[j], kz[j], ok = krumning(sec, -N, My[j], Mz[j])
            if not ok:
                if it == 1:
                    return pakk(False, "materialbrudd",
                                "Tverrsnittet tar ikke opp 1. ordens last ved N_Ed.", it)
                return pakk(False, "instabilitet",
                            "Utbøyningen vokste til brudd – søylen er for slank.", it)

        a = s.relax
        vy = (1 - a) * vy + a * integrer(ky, dx, s.boundary)
        vz = (1 - a) * vz + a * integrer(kz, dx, s.boundary)

        e2 = float(np.max(np.hypot(vy, vz)))
        M2 = N * e2
        hist.append(M2)
        if verbose:
            print(f"  iter {it:2d}:  e2 = {e2:7.3f} mm   M2 = {M2/1e6:8.3f} kNm")

        if M2_prev is not None and abs(M2 - M2_prev) / max(abs(M2), 1e-9) < s.tol_M2:
            return pakk(True, "stabil", "Likevekt funnet – søylen er stabil.", it)
        if e2 > L / 10:
            return pakk(False, "instabilitet", "Utbøyningen løper løpsk – instabilitet.", it)
        M2_prev = M2

    return pakk(False, "instabilitet", "Ikke konvergert – sannsynlig instabilitet.", s.max_iter)


# --- kapasitetssøk -------------------------------------------------
def kapasitetsfaktor(s: Soyle, hi=5.0, tol=0.01) -> float:
    """Største lastfaktor der søylen holder. <1 betyr svikt under dim. last."""
    import copy

    def holder(beta):
        s2 = copy.copy(s)
        s2.N_Ed = s.N_Ed * beta
        for a in ("M0y_top", "M0y_bot", "M0z_top", "M0z_bot", "qy", "Qy", "qz", "Qz"):
            setattr(s2, a, getattr(s, a) * beta)
        r = generell_metode(s2)
        return r.stabil and r.utn <= 1.0

    if holder(hi):
        return hi
    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if holder(mid) else (lo, mid)
    return lo


# --- forenklede metoder 5.8.7 og 5.8.8 -----------------------------
def M0Ed_ekv(s: Soyle):
    if abs(s.M0y_top) >= abs(s.M0y_bot):
        M02, M01 = s.M0y_top, s.M0y_bot
    else:
        M02, M01 = s.M0y_bot, s.M0y_top
    M0e = max(0.6 * abs(M02) + 0.4 * abs(M01), 0.4 * abs(M02))
    return M0e + abs(s.N_Ed) * e_imperf(s)


def nominell_krumning(s: Soyle, c=10.0):
    phi = phi_ef(s)
    sec = bygg_tverrsnitt(s, phi)
    Ac, i, As, d = tverrsnitt_data(s)
    lam, _, n, fcd = slankhet(s, phi)
    fyd = s.fyk / 1.15
    omega = As * fyd / (Ac * fcd)
    Kr = max(0.0, min(1.0, (1 + omega - n) / (1 + omega - 0.4)))
    beta = 0.35 + s.fck / 200 - lam / 150
    Kphi = max(1.0, 1 + beta * phi)
    e2 = Kr * Kphi * (fyd / s.Es) / (0.45 * d) * s.l0 ** 2 / c
    M_Ed = M0Ed_ekv(s) + abs(s.N_Ed) * e2
    cap = M_Rd(sec, -abs(s.N_Ed))
    return {"e2": e2, "M_Ed": M_Ed, "M_Rd": cap, "utn": M_Ed / cap}


def nominell_stivhet(s: Soyle):
    phi = phi_ef(s)
    sec = bygg_tverrsnitt(s, phi)
    Ac, i, As, d = tverrsnitt_data(s)
    lam, _, n, fcd = slankhet(s, phi)
    Ecd = create_concrete(fck=s.fck, design_code="ec2_2004").Ecm / 1.2
    if s.section_type == "circle":
        Ic, Is = math.pi * s.D ** 4 / 64, As * (s.D / 2 - s.cover_c) ** 2 / 2
    else:
        Ic, Is = s.b * s.h ** 3 / 12, As * (s.h / 2 - s.cover_c) ** 2
    Kc = math.sqrt(s.fck / 20) * min(n * lam / 170, 0.20) / (1 + phi)
    Ks = 1.0 if As / Ac >= 0.002 else 0.0
    EI = Kc * Ecd * Ic + Ks * s.Es * Is
    NB = math.pi ** 2 * EI / s.l0 ** 2
    M0 = M0Ed_ekv(s)
    M_Ed = M0 * (1 + (math.pi ** 2 / 8) / (NB / abs(s.N_Ed) - 1)) if NB > abs(s.N_Ed) else float("inf")
    cap = M_Rd(sec, -abs(s.N_Ed))
    return {"M_Ed": M_Ed, "M_Rd": cap, "utn": M_Ed / cap if cap > 0 else float("inf")}


# --- plott ---------------------------------------------------------
def plott(r: Resultat, filnavn="soyle_resultat.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xm = r.x / 1000
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Generell metode (5.8.6)", fontsize=13, weight="bold")
    ax[0, 0].plot(np.hypot(r.My, r.Mz) / 1e6, xm, "C0")
    ax[0, 0].set(xlabel="|M(x)| [kNm]", ylabel="x [m]", title="Totalmoment")
    ax[0, 1].plot(np.hypot(r.vy, r.vz), xm, "C1")
    ax[0, 1].set(xlabel="v(x) [mm]", ylabel="x [m]", title="Utbøyning")
    ax[1, 0].plot(np.hypot(r.ky, r.kz) * 1e6, xm, "C2")
    ax[1, 0].set(xlabel="κ(x) [1e-6/mm]", ylabel="x [m]", title="Krumning")
    if r.M2_hist:
        ax[1, 1].plot(range(1, len(r.M2_hist) + 1), [m / 1e6 for m in r.M2_hist], "C3o-")
    ax[1, 1].set(xlabel="iterasjon", ylabel="M2 [kNm]", title="Konvergens")
    for a in ax.flat:
        a.grid(alpha=.3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filnavn, dpi=130)
    plt.close(fig)
    return filnavn


# --- rapport -------------------------------------------------------
def rapport(s: Soyle, lag_plott=False):
    phi = phi_ef(s)
    lam, lam_lim, n, _ = slankhet(s, phi)
    ts = (f"Ø{s.D:.0f}, {s.n_bars_circle}Ø{s.bar_dia:.0f}" if s.section_type == "circle"
          else f"{s.b:.0f}x{s.h:.0f}, 2x{s.n_bars_per_side}+2x{s.n_bars_side}Ø{s.bar_dia:.0f}")

    print("=" * 66)
    print(f" Slank betongsøyle – NS-EN 1992-1-1    ({s.boundary})")
    print("=" * 66)
    print(f" {ts}, C{s.fck:.0f} | l0 = {s.l0/1000:.2f} m | N_Ed = {s.N_Ed/1e3:.0f} kN")
    print(f" M0y = {s.M0y_top/1e6:.0f}/{s.M0y_bot/1e6:.0f} kNm"
          + (f", M0z = {s.M0z_top/1e6:.0f}/{s.M0z_bot/1e6:.0f} kNm" if (s.M0z_top or s.M0z_bot) else "")
          + f" | phi_ef = {phi:.2f}" + (" (auto)" if s.auto_creep else ""))
    print(f" Slankhet lambda = {lam:.1f} (grense {lam_lim:.1f}), n = {n:.2f}"
          + ("  -> 2. ordens må med" if lam > lam_lim else ""))
    print("-" * 66)

    r = generell_metode(s, verbose=True)
    nk = nominell_krumning(s)
    ns = nominell_stivhet(s)

    def st(u, ok=True):
        return "OK" if ok and u <= 1 else "IKKE OK"

    def num(v):
        return f"{v/1e6:.1f}" if math.isfinite(v) else "inf"

    print("-" * 66)
    print(f" {'':<22}{'5.8.6':>12}{'5.8.7':>12}{'5.8.8':>12}")
    print(f" {'e2 [mm]':<22}{r.e2:>12.1f}{'-':>12}{nk['e2']:>12.1f}")
    print(f" {'M_Ed [kNm]':<22}{r.M_Ed/1e6:>12.1f}{num(ns['M_Ed']):>12}{nk['M_Ed']/1e6:>12.1f}")
    print(f" {'M_Rd [kNm]':<22}{r.M_Rd/1e6:>12.1f}{num(ns['M_Rd']):>12}{nk['M_Rd']/1e6:>12.1f}")
    print(f" {'Utnyttelse':<22}{r.utn:>12.2f}{ns['utn']:>12.2f}{nk['utn']:>12.2f}")
    print(f" {'Status':<22}{st(r.utn, r.stabil):>12}{st(ns['utn']):>12}{st(nk['utn']):>12}")
    print("-" * 66)
    print(f" e_i = {r.e_i:.1f} mm | kritisk snitt x = {r.x_krit/1000:.2f} m | {r.iter} iter")
    print(f" {r.aarsak}  (beta = {kapasitetsfaktor(s):.2f})")
    if lag_plott:
        print(f" Plott: {plott(r)}")
    print("=" * 66)
    return r


def batch(soyler, navn=None):
    print(f" {'#':<4}{'tverrsnitt':<12}{'l0':>6}{'N':>7}{'e2':>8}{'M_Ed':>8}{'M_Rd':>8}{'utn':>6}{'beta':>6}  status")
    print("-" * 74)
    for i, s in enumerate(soyler):
        r = generell_metode(s)
        ts = f"Ø{s.D:.0f}" if s.section_type == "circle" else f"{s.b:.0f}x{s.h:.0f}"
        nm = navn[i] if navn else f"S{i+1}"
        ok = "OK" if (r.stabil and r.utn <= 1) else "IKKE OK"
        print(f" {nm:<4}{ts:<12}{s.l0/1000:>6.1f}{s.N_Ed/1e3:>7.0f}{r.e2:>8.1f}"
              f"{r.M_Ed/1e6:>8.1f}{r.M_Rd/1e6:>8.1f}{r.utn:>6.2f}"
              f"{kapasitetsfaktor(s):>6.2f}  {ok}")


if __name__ == "__main__":
    s = Soyle(b=400, h=400, l0=6000, fck=35, n_bars_per_side=4, bar_dia=25,
              N_Ed=2500e3, M0y_top=80e6, M0y_bot=40e6, phi_ef=1.0)
    rapport(s, lag_plott=True)
