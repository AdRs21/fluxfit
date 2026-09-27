"""fluxfit - interactive fluxonium spectrum fitting from 2D flux/frequency scans.

Designed for Jupyter with `%matplotlib widget` (ipympl).  Physics via scqubits.

    from fluxfit import Scan, FluxoniumFitter
    app = FluxoniumFitter([Scan("wide", x=coil_V, f=freq_GHz, z=signal)],
                          with_resonator=False)
    app.show()
"""
from __future__ import annotations
import json, time, warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
import scipy.linalg as sla
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.optimize import (linear_sum_assignment, least_squares,
                            differential_evolution, minimize)
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
import scqubits as sq

__all__ = ["Scan", "Trace", "ModelSpec", "SpectrumModel", "FluxoniumFitter", "label_str",
           "significance_map", "ridge_enhance", "detect_traces", "auto_calibration",
           "image_score", "coarse_search", "assign_traces", "refine_fit", "synthetic_scan",
           "offset_name", "x_eff"]

PARAM_NAMES = ["EJ", "EC", "EL", "E_osc", "g", "x0", "slope"]
PARAM_UNITS = {"EJ": "GHz", "EC": "GHz", "EL": "GHz", "E_osc": "GHz", "g": "GHz",
               "x0": "x-unit", "slope": "Phi0 per x-unit"}

def offset_name(scan_name: str) -> str:
    """Name of the per-scan coil offset parameter."""
    return f"dx_{scan_name}"

def x_eff(p: dict, scan_name: str, x) -> np.ndarray:
    """Coil values of one scan mapped onto the common calibration axis.

    Scans recorded at different times can be displaced in coil value by drift, so each
    scan carries its own offset dx and the flux is
    Phi/Phi_0 = slope * (x - dx_scan - x0).  One scan is the reference with dx fixed at
    0, otherwise dx and x0 are degenerate.
    """
    return np.asarray(x, float) - float(p.get(offset_name(scan_name), 0.0))

# --------------------------------------------------------------------------- data

@dataclass
class Scan:
    """One 2D two-tone scan.  `z` may be complex; shape (n_f, n_x) or (n_x, n_f)."""
    name: str
    x: np.ndarray                 # coil value (or flux, see x_unit)
    f: np.ndarray                 # drive frequency
    z: np.ndarray
    x_unit: str = "V"             # "V", "A", "mA", ... or "Phi0"/"flux" if already calibrated
    f_unit: str = "GHz"           # "GHz" or "Hz"

    def __post_init__(self):
        self.x = np.asarray(self.x, float).ravel()
        self.f = np.asarray(self.f, float).ravel()
        u = self.f_unit.lower().replace("hertz", "hz")
        fac = {"ghz": 1.0, "mhz": 1e-3, "khz": 1e-6, "hz": 1e-9}.get(u)
        if fac is None:
            raise ValueError(f"{self.name}: unknown f_unit {self.f_unit!r}; "
                             "use GHz, MHz, kHz or Hz")
        self.f = self.f * fac
        self.f_unit = "GHz"
        if self.f.size and not (1e-3 <= float(np.median(self.f)) <= 1e3):
            warnings.warn(f"{self.name}: the frequency axis spans "
                          f"{self.f[0]:.4g} to {self.f[-1]:.4g} GHz after conversion from "
                          f"{u}; check f_unit, these are not plausible qubit frequencies")
        z = np.asarray(self.z)
        if z.shape == (self.x.size, self.f.size) and z.shape[0] != z.shape[1]:
            z = z.T
        if z.shape != (self.f.size, self.x.size):
            raise ValueError(f"{self.name}: z has shape {z.shape}, expected "
                             f"{(self.f.size, self.x.size)} or its transpose")
        ix, jf = np.argsort(self.x), np.argsort(self.f)
        self.x, self.f, self.z = self.x[ix], self.f[jf], z[np.ix_(jf, ix)]
        for a, nm in ((self.x, "x"), (self.f, "f")):
            d = np.diff(a)
            if a.size > 2 and np.ptp(d) > 0.01 * np.abs(np.median(d)):
                warnings.warn(f"{self.name}: {nm} axis is not uniform "
                              f"(spread {np.ptp(d) / abs(np.median(d)):.1%} of the median step); "
                              "display and smoothing assume a uniform grid")

    @property
    def is_calibrated(self) -> bool:
        return self.x_unit.lower() in ("phi0", "flux", "phi_0", "phi")

    def real(self, component: str = "abs") -> np.ndarray:
        z = self.z
        if not np.iscomplexobj(z):
            return np.asarray(z, float)
        c = component.lower()
        if c == "abs":
            return np.abs(z)
        if c == "logabs":
            return 20 * np.log10(np.abs(z) + 1e-300)
        if c == "re":
            return z.real
        if c == "im":
            return z.imag
        if c == "phase":
            return np.angle(z)
        if c == "dphase":                       # phase gradient along frequency
            ph = np.unwrap(np.angle(z), axis=0)
            return np.gradient(ph, axis=0)
        raise ValueError(f"unknown component {component!r}")

    @property
    def extent(self):
        dx = np.median(np.diff(self.x)) if self.x.size > 1 else 1.0
        df = np.median(np.diff(self.f)) if self.f.size > 1 else 1.0
        return (self.x[0] - dx / 2, self.x[-1] + dx / 2,
                self.f[0] - df / 2, self.f[-1] + df / 2)

# -------------------------------------------------------------------- preprocessing

def significance_map(scan: Scan, component="abs", row_median=False, col_detrend=0.15,
                     smooth_f=1.5, smooth_x=0.0, polarity="auto", norm="global",
                     min_detrend_px=9) -> np.ndarray:
    """Robust z-score map: dimensionless, so one detection threshold works across scans.

    row_median   remove the flux-independent background (horizontal artefacts).  Leave
                 it off for a narrow zoom, where a nearly flat line would be removed too.
    col_detrend  fraction of the frequency span used as the median-filter window that
                 removes the slow baseline of each column.  Disabled automatically when
                 that window would be shorter than min_detrend_px points, since on a
                 short frequency axis it is a high-pass filter that eats the line.
    norm         "global" estimates one noise scale for the whole map; "column"
                 estimates one per flux value.  Per-column normalisation divides a
                 line-dominated column by its own line: on a zoom where the line fills
                 the frequency window the noise estimate can be 90 times larger in the
                 columns containing the line than in the empty ones, which buries it.
                 Use "column" only when the noise really varies with flux.
    polarity     "auto", "positive", "negative" or "abs"
    """
    r = np.asarray(scan.real(component), float)
    r = np.nan_to_num(r, nan=np.nanmedian(r))
    if col_detrend and col_detrend > 0:
        k = int(col_detrend * r.shape[0]) | 1
        if k >= min_detrend_px:
            r = r - ndimage.median_filter(r, size=(k, 1), mode="nearest")
        else:
            warnings.warn(f"{scan.name}: frequency detrending skipped, the window would "
                          f"be {k} points of {r.shape[0]}; on a short frequency axis it "
                          "removes the line together with the baseline")
    if row_median:
        r = r - np.median(r, axis=1, keepdims=True)
    med = np.median(r, axis=0, keepdims=True)
    if norm == "column":
        sig = 1.4826 * np.median(np.abs(r - med), axis=0, keepdims=True)
        sig = np.where(sig <= 0, np.median(sig[sig > 0]) if np.any(sig > 0) else 1.0, sig)
    elif norm == "global":
        sig = 1.4826 * float(np.median(np.abs(r - med)))
        sig = sig if sig > 0 else 1.0
    else:
        raise ValueError(f"norm must be 'global' or 'column', not {norm!r}")
    s = (r - med) / sig
    if polarity == "auto":
        hi, lo = np.percentile(s, 99.5), np.percentile(s, 0.5)
        polarity = "negative" if abs(lo) > abs(hi) else "positive"
    if polarity == "negative":
        s = -s
    elif polarity == "abs":
        s = np.abs(s)
    if smooth_f or smooth_x:
        s = ndimage.gaussian_filter(s, sigma=(smooth_f, smooth_x), mode="nearest")
    return s

def ridge_enhance(s: np.ndarray, width_px: float = 2.0) -> np.ndarray:
    """Meijering neuriteness filter, renormalised back to a z-score-like scale."""
    from skimage.filters import meijering
    sigmas = [max(1.0, width_px * a) for a in (0.5, 1.0, 2.0)]
    e = meijering(s, sigmas=sigmas, black_ridges=False)
    e = np.nan_to_num(e)
    med = np.median(e)
    mad = np.median(np.abs(e - med)) * 1.4826
    return (e - med) / (mad if mad > 0 else 1.0)

# ----------------------------------------------------------------------- detection

@dataclass
class Trace:
    scan: str
    x: np.ndarray
    f: np.ndarray
    w: np.ndarray
    tid: int = 0
    label: Optional[tuple] = None      # (state_a, state_b, n_photon)
    include: bool = True
    source: str = "auto"               # "auto" | "manual"
    confidence: float = np.nan         # 2nd-best / best distance ratio, >1 is good
    residual: float = np.nan           # median |f_meas - f_model| in GHz
    manual_label: bool = False         # set by hand; automatic assignment leaves it alone

    def __len__(self):
        return int(self.x.size)

    def decimate(self, n: int):
        if self.x.size <= n:
            return self.x, self.f, self.w
        k = np.unique(np.linspace(0, self.x.size - 1, n).round().astype(int))
        return self.x[k], self.f[k], self.w[k]

    def to_dict(self):
        d = dict(scan=self.scan, x=self.x.tolist(), f=self.f.tolist(), w=self.w.tolist(),
                 tid=self.tid, include=bool(self.include), source=self.source,
                 confidence=float(self.confidence), residual=float(self.residual),
                 manual_label=bool(self.manual_label))
        d["label"] = None if self.label is None else json.loads(json.dumps(self.label))
        return d

    @staticmethod
    def from_dict(d):
        lab = d.get("label")
        if lab is not None:
            lab = (_as_state(lab[0]), _as_state(lab[1]), int(lab[2]))
        return Trace(scan=d["scan"], x=np.array(d["x"]), f=np.array(d["f"]),
                     w=np.array(d["w"]), tid=d.get("tid", 0), label=lab,
                     include=d.get("include", True), source=d.get("source", "auto"),
                     confidence=d.get("confidence", np.nan), residual=d.get("residual", np.nan),
                     manual_label=d.get("manual_label", False))

def _as_state(s):
    return tuple(s) if isinstance(s, (list, tuple)) else int(s)

def _subpixel(col, i):
    """Parabolic refinement of a peak at integer index i; returns a fractional index."""
    if 0 < i < col.size - 1:
        a, b, c = col[i - 1], col[i], col[i + 1]
        den = a - 2 * b + c
        if den != 0:
            return i + np.clip(0.5 * (a - c) / den, -0.5, 0.5)
    return float(i)

def detect_traces(s: np.ndarray, x: np.ndarray, f: np.ndarray, scan_name="",
                  prominence=3.0, min_length=8, max_gap=2, tol=None, max_peaks=12,
                  min_separation_px=3, ridge=False, ridge_width=2.0,
                  median_smooth=3, start_tid=0) -> list[Trace]:
    """Column-wise peak finding followed by greedy track linking.

    tol : matching tolerance in GHz around the slope-extrapolated prediction.
          Defaults to 6 frequency pixels.
    """
    if ridge:
        s = ridge_enhance(s, ridge_width)
    nf, nx = s.shape
    df = float(np.median(np.diff(f))) if nf > 1 else 1.0
    if tol is None:
        tol = 6 * abs(df)
    cols = []
    for j in range(nx):
        pk, props = find_peaks(s[:, j], prominence=prominence, distance=min_separation_px)
        if pk.size > max_peaks:
            keep = np.argsort(props["prominences"])[-max_peaks:]
            pk, prom = pk[keep], props["prominences"][keep]
        else:
            prom = props["prominences"]
        fr = np.array([np.interp(_subpixel(s[:, j], i), np.arange(nf), f) for i in pk])
        cols.append((fr, prom))
    live, done = [], []
    for j in range(nx):
        fr, prom = cols[j]
        pred = []
        for tr in live:
            if len(tr["f"]) >= 3:
                k = min(6, len(tr["f"]))
                sl = np.polyfit(tr["x"][-k:], tr["f"][-k:], 1)[0]
            else:
                sl = 0.0
            pred.append(tr["f"][-1] + sl * (x[j] - tr["x"][-1]))
        if live and fr.size:
            cost = np.abs(np.array(pred)[:, None] - fr[None, :])
            cost[cost > tol] = 1e6
            ri, ci = linear_sum_assignment(cost)
            used = set()
            for a, b in zip(ri, ci):
                if cost[a, b] < 1e6:
                    live[a]["x"].append(x[j]); live[a]["f"].append(fr[b])
                    live[a]["w"].append(prom[b]); live[a]["last"] = j
                    used.add(b)
            new = [b for b in range(fr.size) if b not in used]
        else:
            new = list(range(fr.size))
        for b in new:
            live.append(dict(x=[x[j]], f=[fr[b]], w=[prom[b]], last=j))
        keep = []
        for tr in live:
            (keep if j - tr["last"] <= max_gap else done).append(tr)
        live = keep
    done += live
    out, tid = [], start_tid
    for tr in done:
        if len(tr["x"]) < min_length:
            continue
        fa = np.array(tr["f"])
        if median_smooth and median_smooth > 1 and fa.size > median_smooth:
            fa = ndimage.median_filter(fa, size=int(median_smooth), mode="nearest")
        out.append(Trace(scan=scan_name, x=np.array(tr["x"]), f=fa, w=np.array(tr["w"]), tid=tid))
        tid += 1
    return out

def snap_to_ridge(s, x, f, xs, fs, window_ghz=0.05, median_smooth=3):
    """Take a few clicked points, interpolate, then snap each column to the local maximum."""
    o = np.argsort(xs)
    xs, fs = np.asarray(xs, float)[o], np.asarray(fs, float)[o]
    if xs.size < 2:
        return None
    guess = PchipInterpolator(xs, fs, extrapolate=False) if xs.size > 2 else \
        (lambda t: np.interp(t, xs, fs, left=np.nan, right=np.nan))
    j0, j1 = np.searchsorted(x, xs[0]), np.searchsorted(x, xs[-1])
    cols = np.arange(max(0, j0), min(x.size, j1 + 1))
    if cols.size < 2:
        return None
    fg = np.asarray(guess(x[cols]), float)
    df = float(np.median(np.diff(f)))
    half = max(1, int(round(window_ghz / abs(df))))
    fo, wo, xo = [], [], []
    for j, fc in zip(cols, fg):
        if not np.isfinite(fc):
            continue
        i0 = int(np.clip(np.searchsorted(f, fc) - half, 0, f.size - 1))
        i1 = int(np.clip(i0 + 2 * half + 1, 1, f.size))
        seg = s[i0:i1, j]
        i = i0 + int(np.argmax(seg))
        fo.append(np.interp(_subpixel(s[:, j], i), np.arange(f.size), f))
        wo.append(float(s[i, j])); xo.append(x[j])
    if len(xo) < 2:
        return None
    fo = np.array(fo)
    if median_smooth > 1 and fo.size > median_smooth:
        fo = ndimage.median_filter(fo, size=int(median_smooth), mode="nearest")
    return np.array(xo), fo, np.array(wo)

# --------------------------------------------------------------------- calibration

def auto_calibration(s: np.ndarray, x: np.ndarray, min_period_px=8):
    """Find the flux period and the two symmetry axes (Phi=0 and Phi=1/2) from the image.

    Model-free: uses the periodicity and the mirror symmetry of the spectrum only.
    Returns a dict with period (x units), the candidate symmetry centres, and the
    proposed (x0, slope).  The assignment of a centre to Phi=0 rather than Phi=1/2 uses
    the spectral centroid (lowest transition frequency at half flux) and can be wrong;
    the UI exposes a flip button.
    """
    c = s - s.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(c, axis=0)
    c = c / np.where(nrm > 0, nrm, 1.0)
    nx = x.size
    dx = float(np.median(np.diff(x)))
    kmax = max(min_period_px + 1, nx - 2)
    ac = np.array([np.mean(np.sum(c[:, :nx - k] * c[:, k:], axis=0)) for k in range(kmax)])
    kk = np.arange(min_period_px, kmax)
    period_px = float(kk[np.argmax(ac[min_period_px:kmax])]) if kk.size else float(nx)
    half = int(round(period_px / 4))
    centres, score = [], np.full(nx, -np.inf)
    for ci in range(nx):
        h = min(ci, nx - 1 - ci, max(3, half))
        if h < 3:
            continue
        d = np.arange(1, h + 1)
        score[ci] = np.mean(np.sum(c[:, ci - d] * c[:, ci + d], axis=0))
    ok = np.isfinite(score)
    if ok.sum() > 3:
        pk, _ = find_peaks(np.where(ok, score, -1e9), distance=max(3, int(period_px / 4)))
        pk = pk[np.argsort(score[pk])[::-1]][:6]
        centres = sorted(pk.tolist())
    if not centres:
        return dict(period=period_px * dx, centres=[], x0=float(x[nx // 2]),
                    slope=1.0 / (period_px * dx), confidence=0.0)
    best = int(max(centres, key=lambda i: score[i]))
    if len(centres) > 1:
        gaps = [abs(x[c2] - x[best]) for c2 in centres if c2 != best]
        period = 2 * min(gaps)
    else:
        period = period_px * dx
    # a centre is half flux when the visible spectral weight sits lower in frequency
    w = np.clip(s, 0, None)
    cen_freq = {}
    for ci in centres:
        sl = slice(max(0, ci - 3), min(nx, ci + 4))
        ww = w[:, sl].sum(axis=1)
        cen_freq[ci] = float(np.sum(ww * np.arange(ww.size)) / max(ww.sum(), 1e-12))
    half_flux = min(cen_freq, key=cen_freq.get)
    x_half = float(x[half_flux])
    slope = 1.0 / period
    x0 = x_half - 0.5 / slope
    srt = np.sort(score[np.isfinite(score)])
    conf = float((score[best] - np.median(srt)) / (np.std(srt) + 1e-12))
    return dict(period=float(period), centres=[float(x[i]) for i in centres],
                x0=float(x0), slope=float(slope), confidence=conf,
                x_half=x_half, x_zero=x0)

# --------------------------------------------------------------------------- model

@dataclass
class ModelSpec:
    with_resonator: bool = False
    n_fluxonium: int = 7          # fluxonium levels kept in the coupled space
    n_resonator: int = 5
    cutoff: int = 110             # fluxonium oscillator-basis cutoff (scqubits)
    from_levels: tuple = (0, 1)   # initial levels (thermal population of |1> is common)
    max_photon: int = 1           # 1 = single photon only, 2 adds f/2 lines
    matelem_min: float = 0.02     # |<i|n|j>| threshold for a line to be drawn/scored

_OPCACHE: dict = {}

def _fluxonium_ops(EC, EL, cutoff):
    """cos(phi), sin(phi), n and the LC diagonal from scqubits, cached over flux.

    scqubits rebuilds the whole Hamiltonian at each flux value; since flux enters only
    through cos(phi + 2 pi f), caching these matrices reproduces
    Fluxonium.hamiltonian() to 1.4e-14 GHz while running about 9 times faster.
    """
    key = (round(float(EC), 12), round(float(EL), 12), int(cutoff))
    hit = _OPCACHE.get(key)
    if hit is None:
        fl = sq.Fluxonium(EJ=1.0, EC=EC, EL=EL, flux=0.0, cutoff=cutoff)
        lc = np.array([(i + 0.5) * fl.plasma_energy() for i in range(cutoff)])
        hit = (lc, np.asarray(fl.cos_phi_operator(beta=0.0)),
               np.asarray(fl.sin_phi_operator(beta=0.0)), np.asarray(fl.n_operator()))
        if len(_OPCACHE) > 8:
            _OPCACHE.pop(next(iter(_OPCACHE)))
        _OPCACHE[key] = hit
    return hit

def fold_flux(phi):
    """Spectrum is 1-periodic and even in flux: fold onto [0, 1/2]."""
    t = np.mod(np.asarray(phi, float), 1.0)
    return np.minimum(t, 1.0 - t)

class SpectrumModel:
    """Transition frequencies of a fluxonium, optionally coupled to one oscillator."""

    def __init__(self, spec: ModelSpec = None):
        self.spec = spec or ModelSpec()

    # -- bare fluxonium -----------------------------------------------------
    def _bare(self, EJ, EC, EL, phis):
        sp = self.spec
        nq = sp.n_fluxonium
        lc, cos0, sin0, nop = _fluxonium_ops(EC, EL, sp.cutoff)
        E = np.empty((phis.size, nq))
        N = np.empty((phis.size, nq, nq), complex)
        for k, ph in enumerate(phis):
            H = -EJ * (cos0 * np.cos(2 * np.pi * ph) - sin0 * np.sin(2 * np.pi * ph))
            H[np.diag_indices_from(H)] += lc
            w, v = sla.eigh(H, subset_by_index=[0, nq - 1])
            E[k] = w - w[0]
            N[k] = v.conj().T @ nop @ v
        return E, N

    def levels(self, p: dict, phis: np.ndarray):
        """Returns (energies, n-matrix-elements, labels) for the folded flux values."""
        sp = self.spec
        Eq, Nq = self._bare(p["EJ"], p["EC"], p["EL"], phis)
        if not sp.with_resonator:
            labels = [(i,) for i in range(sp.n_fluxonium)]
            return Eq, np.abs(Nq), labels
        nq, nr = sp.n_fluxonium, sp.n_resonator
        a = np.diag(np.sqrt(np.arange(1, nr)), 1)
        xr = a + a.T
        Hr = np.diag(p["E_osc"] * np.arange(nr))
        Ir, Iq = np.eye(nr), np.eye(nq)
        dim = nq * nr
        E = np.empty((phis.size, dim)); W = np.empty((phis.size, dim, dim))
        lab = np.empty((phis.size, dim, 2), int)
        nq_full = np.empty((dim, dim), complex)
        for k in range(phis.size):
            H = np.kron(np.diag(Eq[k]), Ir) + np.kron(Iq, Hr) + p["g"] * np.kron(Nq[k], xr)
            w, v = sla.eigh(H)
            E[k] = w - w[0]
            nq_full[:] = np.kron(Nq[k], Ir)
            W[k] = np.abs(v.conj().T @ nq_full @ v)
            idx = np.argmax(np.abs(v) ** 2, axis=0)
            lab[k, :, 0], lab[k, :, 1] = idx // nr, idx % nr
        return E, W, lab

    # -- transitions --------------------------------------------------------
    def branches(self, p: dict, x: np.ndarray, f_window=None):
        """Predicted transition frequencies on the x grid.

        Returns {label: (freqs, weight)} with label = (state_a, state_b, n_photon);
        states are ints (bare fluxonium) or (q, n) tuples (coupled system).
        """
        sp = self.spec
        phi = fold_flux(p["slope"] * (np.asarray(x, float) - p["x0"]))
        uphi, inv = np.unique(np.round(phi, 9), return_inverse=True)
        out = {}
        if not sp.with_resonator:
            E, W, _ = self.levels(p, uphi)
            for i in sp.from_levels:
                if i >= sp.n_fluxonium:
                    continue
                for j in range(i + 1, sp.n_fluxonium):
                    w = float(np.median(W[:, i, j]))
                    if w < sp.matelem_min:
                        continue
                    fr = (E[:, j] - E[:, i])[inv]
                    for nph in range(1, sp.max_photon + 1):
                        out[(i, j, nph)] = (fr / nph, w * (0.3 ** (nph - 1)))
        else:
            E, W, lab = self.levels(p, uphi)
            dim = E.shape[1]
            ref = min(range(uphi.size), key=lambda k: abs(uphi[k] - np.median(uphi)))
            init = [k for k in range(dim) if tuple(lab[ref, k]) in
                    [(i, 0) for i in sp.from_levels]]
            for a in init:
                la = tuple(lab[ref, a])
                for b in range(dim):
                    lb = tuple(lab[ref, b])
                    if b == a or E[ref, b] <= E[ref, a]:
                        continue
                    w = float(np.median(W[:, a, b]))
                    if w < sp.matelem_min:
                        continue
                    fr = np.full(uphi.size, np.nan)
                    for k in range(uphi.size):     # follow the labels across flux
                        ka = np.flatnonzero((lab[k, :, 0] == la[0]) & (lab[k, :, 1] == la[1]))
                        kb = np.flatnonzero((lab[k, :, 0] == lb[0]) & (lab[k, :, 1] == lb[1]))
                        if ka.size and kb.size:
                            fr[k] = E[k, kb[0]] - E[k, ka[0]]
                    fr = fr[inv]
                    for nph in range(1, sp.max_photon + 1):
                        out[(la, lb, nph)] = (fr / nph, w * (0.3 ** (nph - 1)))
        if f_window is not None:
            lo, hi = f_window
            out = {k: v for k, v in out.items()
                   if np.any((v[0] >= lo) & (v[0] <= hi))}
        return out

def label_str(lab) -> str:
    a, b, nph = lab
    s = (f"{a}-{b}" if isinstance(a, int) else
         f"|{a[0]},{a[1]}>-|{b[0]},{b[1]}>")
    return s + ("" if nph == 1 else f" /{nph}ph")

# ------------------------------------------------------------------------- fitting

W_SAT = 0.3     # |<i|n|j>| above which a line is counted as fully visible

def _interp(s, scan):
    return RegularGridInterpolator((scan.f, scan.x), s, bounds_error=False, fill_value=0.0)

def blur_maps(maps: dict, scans: Sequence[Scan], sigma_ghz: float, n_flux: int = None) -> dict:
    """Gaussian blur of the significance map, renormalised to a z-score scale.

    Along frequency, widening the lines widens the basin of attraction of the coarse
    search: with 10 MHz lines a 1 % error on the flux period already misses them.

    Along flux, the blur removes the aliasing of the score.  The score samples each
    predicted line at n_flux positions; on a scan of 500 columns with sharp lines, a
    sample either lands on a line pixel or misses it, which made the score vary by a
    factor of 7 between n_flux = 21 and n_flux = 25 on real data and let the optimiser
    chase particular sampling alignments.  Blurring by half the sampling step makes each
    sample represent its neighbourhood, and the score a smooth function of the
    parameters.
    """
    out = {}
    for sc in scans:
        b = maps[sc.name]
        sx = 0.0 if not n_flux else max(0.8, 0.5 * sc.x.size / n_flux)
        if sigma_ghz:
            df = abs(float(np.median(np.diff(sc.f))))
            b = ndimage.gaussian_filter1d(b, sigma_ghz / df, axis=0, mode="nearest")
        if sx:
            b = ndimage.gaussian_filter1d(b, sx, axis=1, mode="nearest")
        if sigma_ghz or sx:
            med = np.median(b); mad = np.median(np.abs(b - med)) * 1.4826
            b = (b - med) / (mad if mad > 0 else 1.0)
        out[sc.name] = b
    return out

def image_score(model: SpectrumModel, p: dict, scans: Sequence[Scan],
                maps: dict, lam=0.6, n_flux=25, interps: dict = None) -> float:
    """Assignment-free score: significance integrated along every predicted line.

    Each sampled point of a predicted line earns S (the robust z-score) and costs lam,
    so a line sitting on background is a net loss while a line outside the measured
    window is neutral.  No detected traces and no labels are needed, which is what makes
    the coarse search work on faint lines.
    """
    tot = 0.0
    for sc in scans:
        itp = (interps or {}).get(sc.name) or _interp(maps[sc.name], sc)
        xs = np.linspace(sc.x[0], sc.x[-1], n_flux)
        br = model.branches(p, x_eff(p, sc.name, xs), f_window=(sc.f[0], sc.f[-1]))
        for lab, (fr, w) in br.items():
            m = np.isfinite(fr) & (fr >= sc.f[0]) & (fr <= sc.f[-1])
            if not m.any():
                continue
            v = itp(np.column_stack([fr[m], xs[m]]))
            tot += min(w / W_SAT, 1.0) * float(np.sum(v - lam))
    return tot / (n_flux * max(1, len(scans)))

def coarse_search(model: SpectrumModel, p0: dict, bounds: dict, free: Sequence[str],
                  scans: Sequence[Scan], maps: dict, lam=0.6, n_flux=25, maxiter=40,
                  popsize=12, seed=0, callback=None, coarse_spec: ModelSpec = None,
                  blur=(0.08, 0.03, 0.0)):
    """Global search over the free parameters, maximising image_score.

    Multi-resolution: differential evolution on the most blurred image, then a local
    Nelder-Mead polish at each sharper scale.  blur values are in GHz.
    """
    free = [k for k in free if k in bounds]
    if not free:
        raise ValueError("no free parameters for the coarse search")
    m = SpectrumModel(coarse_spec) if coarse_spec is not None else model
    n_eval = [0]
    t0 = time.time()

    def make_neg(mp, names):
        itp = {sc.name: _interp(mp[sc.name], sc) for sc in scans}

        def neg(v):
            p = dict(p0); p.update({k: float(val) for k, val in zip(names, v)})
            n_eval[0] += 1
            try:
                s = image_score(m, p, scans, mp, lam=lam, n_flux=n_flux, interps=itp)
            except Exception:
                return 1e6
            if callback and n_eval[0] % 25 == 0:
                callback(n_eval[0], s, time.time() - t0)
            return -s
        return neg

    # the global stage runs over the energies only, with the flux calibration held at its
    # symmetry-derived value: a 5-dimensional rugged landscape needs far more evaluations
    # than a 3-dimensional one, and x0/slope are already known to a few % of a period.
    core = [k for k in free
            if k not in ("x0", "slope") and not k.startswith("dx_")] or list(free)
    sub = list(free)
    keep = dict(p0)

    def embed(v_core):
        p = dict(keep); p.update({k: float(x) for k, x in zip(core, v_core)})
        return [p[k] for k in sub]

    res = differential_evolution(make_neg(blur_maps(maps, scans, blur[0], n_flux), core),
                                 [bounds[k] for k in core], maxiter=maxiter,
                                 popsize=popsize, seed=seed, polish=True, tol=1e-4,
                                 mutation=(0.4, 1.2), recombination=0.8, init="sobol")
    best, score = np.array(embed(res.x), float), -res.fun
    lo = np.array([bounds[k][0] for k in free]); hi = np.array([bounds[k][1] for k in free])
    for b in blur[1:]:
        neg = make_neg(blur_maps(maps, scans, b, n_flux), free)
        r = minimize(neg, best, method="Nelder-Mead",
                     options=dict(xatol=1e-6, fatol=1e-5, maxiter=600 * len(free)))
        best, score = np.clip(r.x, lo, hi), -r.fun
    p = dict(p0); p.update({k: float(v) for k, v in zip(free, best)})
    return p, dict(score=float(score), n_eval=n_eval[0], seconds=time.time() - t0,
                   message=str(res.message))

def _gather(traces, n_pts):
    """Concatenate the decimated points of several traces, keeping their scan of origin."""
    xs, slices, ids, i = [], [], [], 0
    names = sorted({tr.scan for tr in traces})
    for tr in traces:
        xd, fd, wd = tr.decimate(n_pts)
        xs.append(xd); slices.append((i, i + xd.size, fd, wd))
        ids.append(np.full(xd.size, names.index(tr.scan))); i += xd.size
    return ((np.concatenate(xs) if xs else np.zeros(0)), slices,
            (np.concatenate(ids) if ids else np.zeros(0, int)), names)

def _apply_offsets(p, xall, ids, names):
    off = np.array([float(p.get(offset_name(n), 0.0)) for n in names])
    return xall - off[ids] if off.size else xall

def assign_traces(model: SpectrumModel, p: dict, traces: Sequence[Trace],
                  tol=0.05, n_pts=25, only_included=True, respect_manual=True):
    """Label each trace by the closest predicted branch (median distance).

    Traces labelled by hand keep their label unless respect_manual is False.
    """
    sel = [t for t in traces if (t.include or not only_included)
           and not (respect_manual and t.manual_label and t.label is not None)]
    if not sel:
        return 0
    xall, sl, ids, names = _gather(sel, n_pts)
    br = model.branches(p, _apply_offsets(p, xall, ids, names))
    n_ok = 0
    for tr, (i0, i1, fd, _) in zip(sel, sl):
        best, second, blab = np.inf, np.inf, None
        for lab, (fr, w) in br.items():
            seg = fr[i0:i1]
            m = np.isfinite(seg)
            if m.sum() < max(3, 0.5 * seg.size):
                continue
            d = float(np.median(np.abs(fd[m] - seg[m])))
            if d < best:
                best, second, blab = d, best, lab
            elif d < second:
                second = d
        if blab is not None and best <= tol:
            tr.label, tr.residual, n_ok = blab, best, n_ok + 1
            tr.confidence = float(second / best) if np.isfinite(second) and best > 0 else np.inf
        else:
            tr.label, tr.residual, tr.confidence = None, best, np.nan
    return n_ok

def refine_fit(model: SpectrumModel, p0: dict, traces: Sequence[Trace], free: Sequence[str],
               bounds: dict, n_pts=25, loss="soft_l1", f_scale=0.02, weighted=False,
               max_nfev=200, callback=None):
    """Least-squares refinement on the labelled points.  Returns (params, info)."""
    sel = [t for t in traces if t.include and t.label is not None]
    if not sel:
        raise ValueError("no labelled traces to fit")
    free = [k for k in free if k in bounds]
    if not free:
        raise ValueError("no free parameters")
    xall, sl, ids, names = _gather(sel, n_pts)
    meas = np.concatenate([fd for _, _, fd, _ in sl])
    wts = np.concatenate([np.sqrt(np.clip(wd, 0, None)) for _, _, _, wd in sl]) if weighted \
        else np.ones_like(meas)
    wts = wts / np.mean(wts)
    labels = [t.label for t in sel]
    n_eval = [0]

    def model_f(v):
        p = dict(p0); p.update({k: float(val) for k, val in zip(free, v)})
        br = model.branches(p, _apply_offsets(p, xall, ids, names))
        out = np.empty_like(meas)
        for lab, (i0, i1, fd, _) in zip(labels, sl):
            fr = br.get(lab)
            seg = fd if fr is None else np.where(np.isfinite(fr[0][i0:i1]), fr[0][i0:i1], fd)
            out[i0:i1] = seg
        return out

    def resid(v):
        n_eval[0] += 1
        r = (meas - model_f(v)) * wts
        if callback and n_eval[0] % 10 == 0:
            callback(n_eval[0], float(np.sqrt(np.mean(r ** 2))))
        return r

    x0 = np.array([p0[k] for k in free], float)
    lo = np.array([bounds[k][0] for k in free]); hi = np.array([bounds[k][1] for k in free])
    x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)
    t0 = time.time()
    r = least_squares(resid, x0, bounds=(lo, hi), loss=loss, f_scale=f_scale,
                      x_scale="jac", diff_step=1e-4, max_nfev=max_nfev)
    p = dict(p0); p.update({k: float(v) for k, v in zip(free, r.x)})
    m, n = r.fun.size, r.x.size
    sigma = {k: np.nan for k in free}
    try:
        _, sv, VT = np.linalg.svd(r.jac, full_matrices=False)
        thr = np.finfo(float).eps * max(r.jac.shape) * sv[0]
        sv_inv = np.array([1 / s if s > thr else 0.0 for s in sv])
        cov = (VT.T * sv_inv ** 2) @ VT * (2 * r.cost / max(m - n, 1))
        for i, k in enumerate(free):
            sigma[k] = float(np.sqrt(abs(cov[i, i])))
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    except Exception:
        cond = np.nan
    rms = float(np.sqrt(np.mean((meas - model_f(r.x)) ** 2)))
    return p, dict(sigma=sigma, rms_GHz=rms, n_points=m, n_free=n, cond=cond,
                   seconds=time.time() - t0, n_eval=n_eval[0], message=str(r.message))

def residual_table(model: SpectrumModel, p: dict, traces: Sequence[Trace], n_pts=25):
    """Per-trace median residual in MHz, for the diagnostics panel."""
    sel = [t for t in traces if t.include and t.label is not None]
    if not sel:
        return []
    xall, sl, ids, names = _gather(sel, n_pts)
    br = model.branches(p, _apply_offsets(p, xall, ids, names))
    rows = []
    for tr, (i0, i1, fd, _) in zip(sel, sl):
        fr = br.get(tr.label)
        if fr is None:
            rows.append((tr, np.nan)); continue
        seg = fr[0][i0:i1]
        m = np.isfinite(seg)
        d = np.median(np.abs(fd[m] - seg[m])) * 1e3 if m.any() else np.nan
        tr.residual = float(d) / 1e3 if np.isfinite(d) else np.nan
        rows.append((tr, float(d)))
    return rows

def synthetic_scan(name, f_range, n_f, x_range, n_x, params: dict, spec: ModelSpec = None,
                   snr=6.0, linewidth=0.012, jitter=0.003, faint=0.4, x_unit="V", seed=0):
    """A fake two-tone scan from a known parameter set, for testing the tool.

    snr        peak height of a fully visible line, in units of the noise standard deviation
    linewidth  Gaussian line width in GHz
    jitter     random-walk wander of each line along frequency, in GHz per column
    faint      fraction of the lines attenuated by a factor 0.35
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(*x_range, n_x); f = np.linspace(*f_range, n_f)
    m = SpectrumModel(spec or ModelSpec())
    img = rng.normal(0, 1.0, (n_f, n_x))
    for lab, (fr, w) in m.branches(params, x, f_window=f_range).items():
        amp = snr * float(np.clip(np.sqrt(w / W_SAT), 0.25, 1.0)) * \
            (0.35 if rng.random() < faint else 1.0)
        jit = np.cumsum(rng.normal(0, jitter, n_x)); jit -= jit.mean()
        ok = np.isfinite(fr)
        img[:, ok] += amp * np.exp(
            -0.5 * ((f[:, None] - (fr[ok] + jit[ok])[None, :]) / linewidth) ** 2)
    img += 0.6 * np.exp(-0.5 * ((f - np.mean(f_range)) /
                                (0.4 * np.ptp(f_range))) ** 2)[:, None]
    return Scan(name, x=x, f=f, z=img, x_unit=x_unit)

# -------------------------------------------------------------------------- the UI

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import ipywidgets as W
from IPython.display import display

_MODES = ["toggle", "label", "select", "draw", "erase", "trim", "calibrate"]
_MODE_HELP = {
    "toggle": "click a line to include/exclude it from the fit",
    "label": "click a line to give it the transition chosen in the box on the right",
    "select": "click a line to select it and see its numbers in the Lines tab",
    "draw": "click a few points along a line, then press 'Finish line'",
    "erase": "click a line to delete it",
    "trim": "click to cut a line at that flux, on the nearer side",
    "calibrate": "click to set the flux reference chosen in the Calibration tab",
}

def _lab_color(lab):
    a, b, nph = lab
    key = (a if isinstance(a, int) else a[0] * 7 + a[1],
           b if isinstance(b, int) else b[0] * 7 + b[1])
    idx = (key[0] * 11 + key[1] * 3 + (nph - 1) * 5) % 20
    return matplotlib.colormaps["tab20"](idx)

def _split_nan(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    segs, i = [], 0
    while i < m.size:
        if m[i]:
            j = i
            while j < m.size and m[j]:
                j += 1
            if j - i > 1:
                segs.append(np.column_stack([x[i:j], y[i:j]]))
            i = j
        else:
            i += 1
    return segs

def _num(value, step, width="90px", **kw):
    return W.FloatText(value=value, step=step, layout=W.Layout(width=width), **kw)

class FluxoniumFitter:
    """Interactive fitter.  Call .show() in a notebook cell after `%matplotlib widget`."""

    def __init__(self, scans: Sequence[Scan], with_resonator=False, params: dict = None,
                 spec: ModelSpec = None, figsize=(6.4, 4.4)):
        if not isinstance(scans, (list, tuple)):
            scans = [scans]
        self.scans = list(scans)
        self.by_name = {s.name: s for s in self.scans}
        if len(self.by_name) != len(self.scans):
            raise ValueError("scan names must be unique")
        self.figsize = figsize
        self.spec = spec or ModelSpec(with_resonator=with_resonator)
        self.model = SpectrumModel(self.spec)
        self.maps: dict = {}
        self.traces: list[Trace] = []
        self._next_tid = 0
        self.selected: Optional[Trace] = None
        self.result = None
        cal = all(s.is_calibrated for s in self.scans)
        xs = np.concatenate([s.x for s in self.scans])
        self.params = dict(EJ=4.0, EC=1.0, EL=0.5, E_osc=7.0, g=0.1,
                           x0=0.0 if cal else float(np.median(xs)),
                           slope=1.0 if cal else 1.0 / max(np.ptp(xs), 1e-9))
        if params:
            self.params.update(params)
        # the calibration is set in the Calibration tab, not by the optimiser: freeing
        # x0 and slope together with the energies diverges unless the labels are already
        # right, so they start fixed and you tick them once the fit is stable
        self.free = dict(EJ=True, EC=True, EL=True, E_osc=True, g=True,
                         x0=False, slope=False)
        self.ref_scan = self.scans[0].name
        self.bounds = dict(EJ=(0.5, 20.0), EC=(0.2, 4.0), EL=(0.05, 3.0),
                           E_osc=(2.0, 12.0), g=(0.0, 0.6),
                           x0=(float(xs.min()) - np.ptp(xs), float(xs.max()) + np.ptp(xs)),
                           slope=(0.02 / max(np.ptp(xs), 1e-9), 20.0 / max(np.ptp(xs), 1e-9)))
        for sc in self.scans[1:]:       # coil drift between scans taken at different times
            k = offset_name(sc.name)
            self.params[k] = 0.0
            self.free[k] = False
            self.bounds[k] = (-0.5 / self.params["slope"], 0.5 / self.params["slope"])
        self._draw_pts: dict = {s.name: [] for s in self.scans}
        self._ui_built = False
        self._silent = False
        if "ipympl" not in matplotlib.get_backend().lower():
            warnings.warn("matplotlib backend is "
                          f"{matplotlib.get_backend()!r}; run `%matplotlib widget` "
                          "in a cell before building the UI, otherwise clicks do nothing")

    def set_spec(self, spec: ModelSpec):
        """Replace the model specification (levels kept, transitions drawn, cutoff)."""
        self.spec = spec
        self.model = SpectrumModel(spec)
        if self._ui_built:
            self._silent = True
            self.w_res.value = spec.with_resonator
            self.w_nq.value = spec.n_fluxonium
            self.w_nr.value = spec.n_resonator
            self.w_cut.value = spec.cutoff
            self.w_from1.value = 1 in spec.from_levels
            self.w_2ph.value = spec.max_photon > 1
            self.w_mel.value = spec.matelem_min
            self._silent = False
            self._draw_model()
        return self

    @property
    def param_names(self):
        """Model parameters plus one coil offset per non-reference scan."""
        return PARAM_NAMES + [offset_name(s.name) for s in self.scans
                              if s.name != self.ref_scan]

    def _free_list(self):
        return [k for k in self.param_names if self.free.get(k)
                and (self.spec.with_resonator or k not in ("E_osc", "g"))]

    # ------------------------------------------------------------- preprocessing
    def preprocess(self):
        kw = dict(component=self.w_component.value, row_median=self.w_rowmed.value,
                  col_detrend=self.w_detrend.value, smooth_f=self.w_smoothf.value,
                  smooth_x=self.w_smoothx.value, polarity=self.w_polarity.value,
                  norm=self.w_norm.value)
        for sc in self.scans:
            self.maps[sc.name] = significance_map(sc, **kw)
        self._log(f"preprocessed {len(self.scans)} scan(s); "
                  + ", ".join(f"{s.name}: max z = {self.maps[s.name].max():.1f}"
                              for s in self.scans))
        for sc in self.scans:
            self._update_image(sc.name)

    def _ensure_maps(self):
        if not self.maps:
            self.preprocess()

    # ---------------------------------------------------------------- detection
    def detect(self, in_view=False):
        self._ensure_maps()
        n0 = len(self.traces)
        for sc in self.scans:
            s, x, f = self.maps[sc.name], sc.x, sc.f
            if in_view:
                ax = self.panels[sc.name]["ax"]
                x0, x1 = sorted(ax.get_xlim()); f0, f1 = sorted(ax.get_ylim())
                jx = np.flatnonzero((x >= x0) & (x <= x1))
                jf = np.flatnonzero((f >= f0) & (f <= f1))
                if jx.size < 4 or jf.size < 4:
                    continue
                s, x, f = s[np.ix_(jf, jx)], x[jx], f[jf]
            new = detect_traces(s, x, f, scan_name=sc.name,
                                prominence=self.w_prom.value, min_length=self.w_minlen.value,
                                max_gap=self.w_maxgap.value,
                                tol=self.w_tol.value / 1e3 if self.w_tol.value > 0 else None,
                                max_peaks=self.w_maxpk.value, ridge=self.w_ridge.value,
                                median_smooth=self.w_medsm.value, start_tid=self._next_tid)
            self._next_tid += len(new)
            self.traces += new
        self._log(f"detected {len(self.traces) - n0} line(s)"
                  + (" in the current view" if in_view else "")
                  + f"; {len(self.traces)} total")
        self._refresh_traces()

    def clear_traces(self, auto_only=True):
        self.traces = [t for t in self.traces if t.source == "manual"] if auto_only else []
        self.selected = None
        self._refresh_traces()

    # -------------------------------------------------------------- calibration
    def _set_bounds(self, k, lo, hi):
        self.bounds[k] = (float(lo), float(hi))
        if self._ui_built:
            self._silent = True
            d = self.p_widgets[k]
            d["lo"].value, d["hi"].value = float(lo), float(hi)
            d["sl"].min = min(lo, d["sl"].value); d["sl"].max = max(hi, d["sl"].value)
            d["sl"].step = (hi - lo) / 400
            self._silent = False

    def auto_calibrate(self, narrow=True):
        self._ensure_maps()
        best, sc_best = None, None
        for sc in self.scans:
            r = auto_calibration(self.maps[sc.name], sc.x)
            if best is None or r["confidence"] > best["confidence"]:
                best, sc_best = r, sc
        if best is None:
            return
        dx = float(self.params.get(offset_name(sc_best.name), 0.0))
        self._set_param("x0", best["x0"] - dx); self._set_param("slope", best["slope"])
        if narrow:      # the coarse search only converges in a narrow flux window
            per = 1.0 / best["slope"]
            self._set_bounds("x0", best["x0"] - 0.15 * per, best["x0"] + 0.15 * per)
            self._set_bounds("slope", 0.85 * best["slope"], 1.18 * best["slope"])
            for sc in self.scans:               # drift is a fraction of a period
                k = offset_name(sc.name)
                if k in self.bounds:
                    self._set_bounds(k, -0.15 * per, 0.15 * per)
        self._log(f"auto-calibration on '{sc_best.name}': period = {best['period']:.4g} "
                  f"{sc_best.x_unit}, Phi=1/2 at x = {best['x_half']:.4g}, "
                  f"symmetry score = {best['confidence']:.1f} sigma. "
                  "Use 'flip 0 <-> 1/2' if the two axes are swapped."
                  + (" Bounds on x0 and slope narrowed to +-15 % of a period."
                     if narrow else ""))
        self._draw_model()

    def _calibrate_click(self, name, x_click):
        """A click calibrates the scan it was made in.

        In the reference scan it moves x0 (or the period); in any other scan it moves
        that scan's own offset dx, so that correcting a drifted zoom does not displace
        the wide scan it was calibrated against.
        """
        what = self.w_calibwhat.value
        ref = name == self.ref_scan
        k = "x0" if ref else offset_name(name)
        if what == "Phi = 0":
            target = x_click if ref else x_click - self.params["x0"]
        elif what.endswith("(keep period)"):
            d = 0.5 / self.params["slope"]
            target = x_click - d if ref else x_click - d - self.params["x0"]
        else:                                   # set the period from this click and x0
            d = x_click - self.params["x0"] - (0 if ref else
                                               self.params.get(offset_name(name), 0.0))
            if abs(d) > 1e-12:
                self._set_param("slope", 0.5 / d)
            self._log(f"calibration from '{name}': slope = {self.params['slope']:.6g} "
                      f"Phi_0 per x-unit, period = {1 / self.params['slope']:.6g}")
            self._draw_model(); return
        if not ref:     # an offset is defined modulo one flux period; take the smallest
            per = 1.0 / self.params["slope"]
            target = target - round(target / per) * per
        self._set_param(k, target)
        if ref:
            self._log(f"calibration: x0 = {self.params['x0']:.6g}, period = "
                      f"{1 / self.params['slope']:.6g} x-units")
        else:
            lo, hi = self.bounds.get(k, (target, target))
            if not (lo <= target <= hi):
                m = 0.2 * max(abs(target), abs(hi - lo)) + 1e-12
                self._set_bounds(k, min(lo, target - m), max(hi, target + m))
            dphi = target * self.params["slope"]
            self._log(f"offset of '{name}': {k} = {target:.6g} x-units "
                      f"({dphi:+.4f} Phi_0 relative to '{self.ref_scan}')")
            if abs(dphi) > 0.25:
                self._log(f"  warning: {abs(dphi):.2f} Phi_0 is not coil drift, it is "
                          "close to half a period.  The reference scan most likely has "
                          "Phi = 0 and Phi = 1/2 swapped: press 'flip 0 <-> 1/2' and "
                          "click again.")
        self._draw_model()

    def set_reference_scan(self, name):
        """Choose which scan defines the common flux axis, keeping every mapping fixed."""
        if name == self.ref_scan:
            return
        shift = float(self.params.get(offset_name(name), 0.0))
        self.params["x0"] = self.params["x0"] + shift
        old_ref = self.ref_scan
        self.ref_scan = name
        for sc in self.scans:
            k = offset_name(sc.name)
            if sc.name == name:
                self.params.pop(k, None); self.free.pop(k, None); self.bounds.pop(k, None)
            else:
                self.params[k] = float(self.params.get(k, 0.0)) - shift
                self.free.setdefault(k, True)
                self.bounds.setdefault(k, (-0.5 / self.params["slope"],
                                           0.5 / self.params["slope"]))
        self._log(f"reference scan is now '{name}' (was '{old_ref}'); "
                  "the flux mapping of every scan is unchanged")
        self._rebuild_model_tab()

    def _rebuild_model_tab(self):
        if not self._ui_built:
            return
        ch = list(self._tab.children)
        ch[3] = self._tab_model()
        self._tab.children = tuple(ch)
        self._tab.set_title(3, "Model")
        self._sync_param_widgets()
        self._draw_model()

    def scale_period(self, factor):
        """Multiply the flux period, keeping the current half-flux point fixed.

        The symmetry axes of a spectrum repeat every half period, so an autocorrelation
        can lock onto P/2 instead of P.  On a synthetic test the auto-calibration
        returned 0.733 units against a true period of 1.429, an alias by a factor 1.95.
        If the model lines are twice too dense or twice too sparse in flux, this is the
        button to press.
        """
        x_half = self.params["x0"] + 0.5 / self.params["slope"]
        self._set_param("slope", self.params["slope"] / factor)
        self._set_param("x0", x_half - 0.5 / self.params["slope"])
        self._log(f"period scaled by {factor}: now {1 / self.params['slope']:.6g} x-units, "
                  f"half flux held at {x_half:.6g}")
        self._draw_model()

    def flip_calibration(self):
        self._set_param("x0", self.params["x0"] + 0.5 / self.params["slope"])
        self._log("flipped the Phi = 0 and Phi = 1/2 assignment")
        self._draw_model()

    # --------------------------------------------------------------------- fits
    def run_coarse(self):
        self._ensure_maps()
        free = self._free_list()
        cs = ModelSpec(**{**self.spec.__dict__, "cutoff": min(self.spec.cutoff, 60),
                          "n_fluxonium": min(self.spec.n_fluxonium, 5),
                          "n_resonator": min(self.spec.n_resonator, 4)})
        self._log(f"coarse search over {free} ...")
        p, info = coarse_search(self.model, self.params, self.bounds, free, self.scans,
                                self.maps, lam=self.w_lam.value, n_flux=self.w_nflux.value,
                                maxiter=self.w_maxiter.value, popsize=self.w_popsize.value,
                                coarse_spec=cs,
                                callback=lambda n, s, t: self._log(
                                    f"  {n} evaluations, best score {s:.2f}, {t:.0f} s",
                                    replace=True))
        for k in free:
            self._set_param(k, p[k])
        self._log(f"coarse search done: score {info['score']:.2f}, {info['n_eval']} "
                  f"evaluations in {info['seconds']:.0f} s")
        self._draw_model()

    def assign(self):
        n = assign_traces(self.model, self.params, self.traces,
                          tol=self.w_asgtol.value / 1e3, n_pts=self.w_npts.value)
        tot = sum(1 for t in self.traces if t.include)
        kept = sum(1 for t in self.traces if t.include and t.manual_label and t.label)
        self._log(f"labelled {n} of {tot - kept} included line(s) within "
                  f"{self.w_asgtol.value:.0f} MHz"
                  + (f"; {kept} hand-set label(s) kept" if kept else ""))
        self._refresh_traces()

    def refine(self):
        free = self._free_list()
        lab = [t for t in self.traces if t.include and t.label is not None]
        n_branch = len({t.label for t in lab})
        self._log(f"fitting {free} on {len(lab)} line(s) carrying {n_branch} distinct "
                  f"transition(s)")
        if not lab:
            self._log("  nothing to fit: no included line carries a label.  Run 'Assign "
                      "labels' first, or raise the assignment tolerance."); return
        if n_branch < len(free):
            self._log(f"  warning: {len(free)} free parameters against {n_branch} "
                      "distinct transition(s).  One transition constrains one "
                      "combination of EJ, EC and EL, not the three separately; label "
                      "more branches or fix some parameters before trusting this.")
        self._log("  running ...")
        try:
            p, info = refine_fit(self.model, self.params, self.traces, free, self.bounds,
                                 n_pts=self.w_npts.value, loss=self.w_loss.value,
                                 f_scale=self.w_fscale.value / 1e3,
                                 weighted=self.w_weighted.value,
                                 callback=lambda n, r: self._log(
                                     f"  {n} residual evaluations, rms {r*1e3:.1f} MHz",
                                     replace=True))
        except ValueError as e:
            self._log(f"cannot fit: {e}"); return
        for k in free:
            self._set_param(k, p[k])
        self.result = info
        self._log(f"fit: rms residual {info['rms_GHz']*1e3:.1f} MHz on {info['n_points']} "
                  f"points, {info['n_free']} free parameters, {info['seconds']:.1f} s")
        if np.isfinite(info["cond"]) and info["cond"] > 1e4:
            self._log(f"  warning: Jacobian condition number {info['cond']:.1e}; "
                      "some parameters are strongly correlated and their uncertainties "
                      "are not independent")
        self._show_result()
        residual_table(self.model, self.params, self.traces, n_pts=self.w_npts.value)
        self._draw_model(); self._refresh_traces()

    def polish(self, rounds=3, tol_start=150.0, tol_end=None):
        """Alternate assignment and refinement with a tightening tolerance.

        A single pass labels only the lines that the starting model already explains: on
        real data the first assignment at 50 MHz caught 3 of 15 traces, while starting
        at 150 MHz and tightening caught 15 of 15 and brought the rms from 15.8 MHz down
        to 13.7 MHz.  The loop is the normal way to use the fit, not a repair.
        """
        t1 = self.w_asgtol.value if tol_end is None else tol_end
        keep = self.w_asgtol.value
        for tol in np.geomspace(max(tol_start, t1), t1, max(1, rounds)):
            self.w_asgtol.value = float(tol)
            self.assign()
            if not any(t.include and t.label is not None for t in self.traces):
                break
            self.refine()
        self.w_asgtol.value = keep

    def guided_run(self):
        self.preprocess()
        if not all(s.is_calibrated for s in self.scans):
            self.auto_calibrate()
        self.run_coarse()
        self.detect()
        self.polish()

    # ------------------------------------------------------------------ session
    def save(self, path):
        d = dict(params=self.params, free=self.free,
                 bounds={k: list(v) for k, v in self.bounds.items()},
                 spec=self.spec.__dict__ | {"from_levels": list(self.spec.from_levels)},
                 traces=[t.to_dict() for t in self.traces],
                 scans=[s.name for s in self.scans], result=self.result)
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1, default=float)
        self._log(f"saved to {path}")

    def load(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.params.update(d["params"]); self.free.update(d["free"])
        self.bounds.update({k: tuple(v) for k, v in d["bounds"].items()})
        sp = dict(d["spec"]); sp["from_levels"] = tuple(sp["from_levels"])
        self.spec = ModelSpec(**sp); self.model = SpectrumModel(self.spec)
        self.traces = [Trace.from_dict(t) for t in d["traces"]]
        self._next_tid = max([t.tid for t in self.traces], default=-1) + 1
        if self._ui_built:
            self._sync_param_widgets(); self._refresh_traces(); self._draw_model()
        self._log(f"loaded {path}: {len(self.traces)} line(s)")
        return self

    # =================================================================== widgets
    def show(self):
        if not self._ui_built:
            self._build()
        display(self.ui)
        self.preprocess()
        self._draw_model()
        return self

    def _build(self):
        self.w_status = W.HTML("")
        self.w_log = W.HTML("", layout=W.Layout(height="88px", overflow="auto",
                                                border="1px solid #ddd", padding="2px"))
        self._log_lines = []
        self.panels = {}
        canvases = []
        for sc in self.scans:
            with plt.ioff():
                fig, ax = plt.subplots(figsize=self.figsize)
            if not isinstance(fig.canvas, W.Widget):
                plt.close(fig)
                raise RuntimeError(
                    "the figure canvas is not an ipywidget (backend "
                    f"{matplotlib.get_backend()!r}).  Run `%matplotlib widget` in a cell "
                    "first; install it with `pip install ipympl` if needed.")
            fig.canvas.header_visible = False
            fig.canvas.toolbar_position = "bottom"
            ax.set_title(sc.name, fontsize=10)
            ax.set_xlabel(f"flux coil ({sc.x_unit})" if not sc.is_calibrated
                          else "flux (Phi_0)")
            ax.set_ylabel("frequency (GHz)")
            im = ax.imshow(np.zeros((sc.f.size, sc.x.size)), origin="lower",
                           extent=sc.extent, aspect="auto", cmap="viridis")
            mcoll = LineCollection([], linewidths=1.2, linestyles="--", zorder=4)
            tcoll = LineCollection([], linewidths=1.6, zorder=5)
            ax.add_collection(mcoll); ax.add_collection(tcoll)
            sel, = ax.plot([], [], color="w", lw=3.0, alpha=0.55, zorder=6)
            clk, = ax.plot([], [], "o", color="w", ms=5, mec="k", zorder=7)
            fig.tight_layout()
            fig.canvas.mpl_connect("button_press_event",
                                   lambda e, n=sc.name: self._on_click(e, n))
            self.panels[sc.name] = dict(fig=fig, ax=ax, im=im, model=mcoll, traces=tcoll,
                                        sel=sel, clicks=clk, texts=[])
            canvases.append(fig.canvas)
        self.w_mode = W.ToggleButtons(options=_MODES, value="toggle",
                                      style={"button_width": "80px"})
        self.w_mode.observe(lambda c: self._set_status(), "value")
        self.w_lab = W.Dropdown(options=[("(none)", None)], description="transition",
                                layout=W.Layout(width="230px"))
        self.ui = W.VBox([W.HBox(canvases),
                          W.HBox([self.w_mode, self.w_lab, self.w_status]),
                          self._tabs(), self.w_log])
        self._ui_built = True
        self._set_status()

    # ---- tabs
    def _tabs(self):
        t = W.Tab(children=[self._tab_view(), self._tab_detect(), self._tab_calib(),
                            self._tab_model(), self._tab_fit(), self._tab_lines()])
        self._tab = t
        for i, n in enumerate(["View", "Detect", "Calibration", "Model", "Fit", "Lines"]):
            t.set_title(i, n)
        return t

    def _tab_view(self):
        self.w_component = W.Dropdown(options=["abs", "logabs", "re", "im", "phase",
                                               "dphase"], value="abs", description="signal",
                                      layout=W.Layout(width="170px"))
        self.w_rowmed = W.Checkbox(value=False, description="remove flux-independent bg",
                                   indent=False)
        self.w_detrend = W.FloatSlider(value=0.15, min=0.0, max=0.5, step=0.01,
                                       description="detrend f", readout_format=".2f")
        self.w_smoothf = W.FloatSlider(value=1.5, min=0.0, max=6.0, step=0.1,
                                       description="smooth f (px)")
        self.w_smoothx = W.FloatSlider(value=0.0, min=0.0, max=4.0, step=0.1,
                                       description="smooth x (px)")
        self.w_polarity = W.Dropdown(options=["auto", "positive", "negative", "abs"],
                                     value="auto", description="peaks",
                                     layout=W.Layout(width="170px"))
        self.w_norm = W.Dropdown(options=["global", "column"], value="global",
                                 description="noise est.", layout=W.Layout(width="180px"))
        self.w_clim = W.FloatRangeSlider(value=[1.0, 99.5], min=0.0, max=100.0, step=0.1,
                                         description="colour %", readout_format=".1f")
        self.w_cmap = W.Dropdown(options=["viridis", "magma", "RdBu_r", "Greys_r"],
                                 value="viridis", description="cmap",
                                 layout=W.Layout(width="170px"))
        self.w_annot = W.Checkbox(value=True, description="label the model lines",
                                  indent=False)
        b = W.Button(description="Apply / preprocess", button_style="info")
        b.on_click(lambda _: self.preprocess())
        for w in (self.w_clim, self.w_cmap):
            w.observe(lambda c: [self._update_image(s.name) for s in self.scans], "value")
        self.w_annot.observe(lambda c: self._draw_model(), "value")
        return W.VBox([W.HBox([self.w_component, self.w_polarity, self.w_norm, self.w_cmap]),
                       W.HBox([self.w_rowmed, self.w_annot]),
                       W.HBox([self.w_detrend, self.w_smoothf, self.w_smoothx]),
                       W.HBox([self.w_clim, b])])

    def _tab_detect(self):
        self.w_prom = W.FloatSlider(value=3.0, min=0.5, max=15.0, step=0.1,
                                    description="threshold (sigma)")
        self.w_minlen = W.IntSlider(value=10, min=3, max=100, description="min length (px)")
        self.w_maxgap = W.IntSlider(value=2, min=0, max=20, description="max gap (px)")
        self.w_tol = W.FloatSlider(value=0.0, min=0.0, max=200.0, step=1.0,
                                   description="link tol (MHz)")
        self.w_maxpk = W.IntSlider(value=12, min=1, max=40, description="peaks / column")
        self.w_medsm = W.IntSlider(value=3, min=1, max=11, step=2, description="smooth line")
        self.w_ridge = W.Checkbox(value=False, description="ridge filter (faint lines)",
                                  indent=False)
        self.w_snapwin = W.FloatSlider(value=50.0, min=5.0, max=300.0, step=5.0,
                                       description="snap (MHz)")
        b1 = W.Button(description="Detect (all)", button_style="info")
        b2 = W.Button(description="Detect in view")
        b3 = W.Button(description="Clear auto lines", button_style="warning")
        b4 = W.Button(description="Finish line", button_style="success")
        b5 = W.Button(description="Cancel points")
        b1.on_click(lambda _: self.detect(False)); b2.on_click(lambda _: self.detect(True))
        b3.on_click(lambda _: self.clear_traces(True))
        b4.on_click(lambda _: self._finish_draw()); b5.on_click(lambda _: self._cancel_draw())
        return W.VBox([W.HBox([self.w_prom, self.w_minlen, self.w_maxgap]),
                       W.HBox([self.w_tol, self.w_maxpk, self.w_medsm]),
                       W.HBox([self.w_ridge, b1, b2, b3]),
                       W.HBox([W.HTML("<b>manual line</b> (mode 'draw'):"), self.w_snapwin,
                               b4, b5])])

    def _tab_calib(self):
        self.w_calibwhat = W.RadioButtons(
            options=["Phi = 0", "Phi = 1/2 (set period)", "Phi = 1/2 (keep period)"],
            value="Phi = 1/2 (keep period)", description="click sets")
        self.w_ref = W.Dropdown(options=[s.name for s in self.scans], value=self.ref_scan,
                                description="reference", layout=W.Layout(width="240px"))
        self.w_ref.observe(lambda c: self.set_reference_scan(c["new"]), "value")
        b1 = W.Button(description="Auto-calibrate", button_style="info")
        b2 = W.Button(description="flip 0 <-> 1/2")
        b1.on_click(lambda _: self.auto_calibrate()); b2.on_click(lambda _: self.flip_calibration())
        b5 = W.Button(description="period x 2", layout=W.Layout(width="110px"))
        b6 = W.Button(description="period / 2", layout=W.Layout(width="110px"))
        b5.on_click(lambda _: self.scale_period(2.0))
        b6.on_click(lambda _: self.scale_period(0.5))
        info = W.HTML("Calibration maps the coil value x to flux: "
                      "Phi/Phi_0 = slope * (x - dx_scan - x0), with one offset dx per "
                      "scan for the coil drift between recordings.  The reference scan "
                      "has dx = 0 by definition; a click calibrates the scan it is made "
                      "in, moving x0 in the reference and dx elsewhere.  Leave the dx "
                      "free in the fit (Model tab) or fix them at 0 for scans taken in "
                      "one run.<br>"
                      "<br>Take the period from the symmetry of a wide scan, then click "
                      "the minimum of the 0-1 line in a low-frequency zoom with "
                      "'Phi = 1/2 (keep period)': that pins half flux far better than the "
                      "symmetry heuristic does.<br>"
                      "Phi/Phi_0 = slope * (x - x0), shared by all scans. "
                      "Set it by clicking, by auto-detection of the mirror symmetry, "
                      "or leave x0 and slope free in the fit (Model tab).")
        return W.VBox([info, W.HBox([self.w_calibwhat,
                                     W.VBox([self.w_ref, b1, b2, W.HBox([b5, b6])])])])

    def _tab_model(self):
        self.w_res = W.Checkbox(value=self.spec.with_resonator, description="couple a resonator",
                                indent=False)
        self.w_nq = W.IntSlider(value=self.spec.n_fluxonium, min=3, max=12,
                                description="fluxonium levels")
        self.w_nr = W.IntSlider(value=self.spec.n_resonator, min=2, max=12,
                                description="resonator levels")
        self.w_cut = W.IntSlider(value=self.spec.cutoff, min=40, max=200, step=10,
                                 description="cutoff")
        self.w_from1 = W.Checkbox(value=1 in self.spec.from_levels, indent=False,
                                  description="also transitions from |1>")
        self.w_2ph = W.Checkbox(value=self.spec.max_photon > 1, indent=False,
                                description="two-photon lines (f/2)")
        self.w_mel = W.FloatSlider(value=self.spec.matelem_min, min=0.0, max=0.5, step=0.005,
                                   description="min |<i|n|j>|", readout_format=".3f")
        self.w_nmodel = W.IntSlider(value=81, min=21, max=301, step=10,
                                    description="model points")
        self.w_live = W.Checkbox(value=False, indent=False,
                                 description="live update while dragging")
        self.p_widgets = {}
        rows = []
        for k in self.param_names:
            lo, hi = self.bounds[k]
            fr = W.Checkbox(value=self.free[k], indent=False, layout=W.Layout(width="26px"))
            va = _num(self.params[k], (hi - lo) / 200, width="100px")
            sl = W.FloatSlider(value=self.params[k], min=lo, max=hi, step=(hi - lo) / 400,
                               readout=False, continuous_update=False,
                               layout=W.Layout(width="230px"))
            wlo, whi = _num(lo, (hi - lo) / 100, width="85px"), _num(hi, (hi - lo) / 100, width="85px")
            self.p_widgets[k] = dict(free=fr, val=va, sl=sl, lo=wlo, hi=whi)
            fr.description_tooltip = k
            fr.observe(lambda c, k=k: self.free.__setitem__(k, c["new"]), "value")
            va.observe(lambda c, k=k: self._param_from_widget(k, c["new"], "val"), "value")
            sl.observe(lambda c, k=k: self._param_from_widget(k, c["new"], "sl"), "value")
            for w in (wlo, whi):
                w.observe(lambda c, k=k: self._bounds_from_widget(k), "value")
            rows.append(W.HBox([fr, W.HTML(f"<code>{k}</code>",
                                           layout=W.Layout(width="95px")), va, sl,
                                W.HTML("bounds"), wlo, whi,
                                W.HTML(f"<i>{PARAM_UNITS.get(k, 'x-unit')}</i>")]))
        for w in (self.w_res, self.w_nq, self.w_nr, self.w_cut, self.w_from1, self.w_2ph):
            w.observe(lambda c: self._spec_changed(), "value")
        self.w_mel.observe(lambda c: self._draw_model(), "value")
        self.w_nmodel.observe(lambda c: self._draw_model(), "value")
        self.w_live.observe(lambda c: [d["sl"].__setattr__("continuous_update", c["new"])
                                       for d in self.p_widgets.values()], "value")
        return W.VBox([W.HBox([self.w_res, self.w_from1, self.w_2ph, self.w_live]),
                       W.HBox([self.w_nq, self.w_nr, self.w_cut]),
                       W.HBox([self.w_mel, self.w_nmodel])] + rows)

    def _tab_fit(self):
        self.w_lam = W.FloatSlider(value=0.6, min=0.0, max=3.0, step=0.05,
                                   description="bg penalty")
        self.w_nflux = W.IntSlider(value=25, min=9, max=81, description="flux points")
        self.w_maxiter = W.IntSlider(value=40, min=5, max=300, description="max iter")
        self.w_popsize = W.IntSlider(value=12, min=4, max=40, description="pop size")
        self.w_asgtol = W.FloatSlider(value=50.0, min=2.0, max=500.0, step=1.0,
                                      description="assign tol (MHz)")
        self.w_npts = W.IntSlider(value=25, min=5, max=100, description="points / line")
        self.w_loss = W.Dropdown(options=["soft_l1", "linear", "huber", "cauchy"],
                                 value="soft_l1", description="loss",
                                 layout=W.Layout(width="180px"))
        self.w_fscale = W.FloatSlider(value=20.0, min=1.0, max=200.0, step=1.0,
                                      description="f_scale (MHz)")
        self.w_weighted = W.Checkbox(value=False, indent=False,
                                     description="weight by line contrast")
        self.w_result = W.HTML("")
        b0 = W.Button(description="Guided run", button_style="success")
        b1 = W.Button(description="Coarse search", button_style="info")
        b2 = W.Button(description="Assign labels")
        b3 = W.Button(description="Refine fit", button_style="info")
        b4 = W.Button(description="Assign + refine loop", button_style="info")
        b4.on_click(lambda _: self.polish())
        b0.on_click(lambda _: self.guided_run()); b1.on_click(lambda _: self.run_coarse())
        b2.on_click(lambda _: self.assign()); b3.on_click(lambda _: self.refine())
        self.w_path = W.Text(value="fluxfit_session.json", description="session",
                             layout=W.Layout(width="320px"))
        bs = W.Button(description="Save"); bl = W.Button(description="Load")
        bs.on_click(lambda _: self.save(self.w_path.value))
        bl.on_click(lambda _: self.load(self.w_path.value))
        return W.VBox([W.HBox([b0, b1, b2, b3, b4]),
                       W.HTML("<b>coarse search</b> (no labels needed)"),
                       W.HBox([self.w_lam, self.w_nflux, self.w_maxiter, self.w_popsize]),
                       W.HTML("<b>assignment and refinement</b>"),
                       W.HBox([self.w_asgtol, self.w_npts, self.w_loss]),
                       W.HBox([self.w_fscale, self.w_weighted]),
                       self.w_result, W.HBox([self.w_path, bs, bl])])

    def _tab_lines(self):
        self.w_lines = W.VBox([])
        b = W.Button(description="apply to selected")
        b.on_click(lambda _: self._apply_label())
        ball = W.Button(description="include all"); bnone = W.Button(description="exclude all")
        ball.on_click(lambda _: self._set_all(True)); bnone.on_click(lambda _: self._set_all(False))
        bdel = W.Button(description="delete unlabelled", button_style="warning")
        bdel.on_click(lambda _: self._del_unlabelled())
        bun = W.Button(description="unlock hand labels")
        bun.on_click(lambda _: self._unlock_labels())
        return W.VBox([W.HBox([b, ball, bnone, bdel, bun]),
                       W.HTML("<small>tick = used in the fit; conf = 2nd best / best "
                              "distance ratio; res = median residual; * = label set by "
                              "hand, which 'Assign labels' will not overwrite</small>"),
                       self.w_lines])

    # ---- widget plumbing
    def _param_from_widget(self, k, v, src):
        if self._silent:
            return
        self.params[k] = float(v)
        self._silent = True
        other = "sl" if src == "val" else "val"
        try:
            self.p_widgets[k][other].value = float(v)
        except Exception:
            pass
        self._silent = False
        self._draw_model()

    def _set_param(self, k, v):
        self.params[k] = float(v)
        if self._ui_built:
            self._silent = True
            d = self.p_widgets[k]
            lo, hi = d["sl"].min, d["sl"].max
            if not (lo <= v <= hi):
                d["sl"].min = min(lo, v - abs(v) * 0.1 - 1e-9)
                d["sl"].max = max(hi, v + abs(v) * 0.1 + 1e-9)
            d["val"].value = float(v); d["sl"].value = float(np.clip(v, d["sl"].min, d["sl"].max))
            self._silent = False

    def _sync_param_widgets(self):
        for k in self.param_names:
            if k in self.p_widgets:
                self._set_param(k, self.params[k])
                self.p_widgets[k]["free"].value = self.free[k]

    def _bounds_from_widget(self, k):
        d = self.p_widgets[k]
        lo, hi = float(d["lo"].value), float(d["hi"].value)
        if hi > lo:
            self.bounds[k] = (lo, hi)
            self._silent = True
            d["sl"].min, d["sl"].max = lo, hi
            d["sl"].step = (hi - lo) / 400
            self._silent = False

    def _spec_changed(self):
        fl = (0, 1) if self.w_from1.value else (0,)
        self.spec = ModelSpec(with_resonator=self.w_res.value, n_fluxonium=self.w_nq.value,
                              n_resonator=self.w_nr.value, cutoff=self.w_cut.value,
                              from_levels=fl, max_photon=2 if self.w_2ph.value else 1,
                              matelem_min=self.w_mel.value)
        self.model = SpectrumModel(self.spec)
        self._draw_model()

    def _log(self, msg, replace=False):
        if not self._ui_built:
            print(msg); return
        if replace and self._log_lines:
            self._log_lines[-1] = msg
        else:
            self._log_lines.append(msg)
        self.w_log.value = "<br>".join(f"<code>{m}</code>" for m in self._log_lines[-40:])

    def _set_status(self, extra=""):
        m = self.w_mode.value
        s = f"<b>{m}</b>: {_MODE_HELP[m]}"
        if self.selected is not None:
            t = self.selected
            s += (f" &nbsp;|&nbsp; selected #{t.tid} ({t.scan}, {len(t)} pts, "
                  f"{label_str(t.label) if t.label else 'unlabelled'})")
        self.w_status.value = s + (f" &nbsp;|&nbsp; {extra}" if extra else "")

    def _show_result(self):
        if not self.result:
            return
        sg = self.result["sigma"]
        rows = ["<tr><th>parameter</th><th>value</th><th>sigma</th><th>unit</th></tr>"]
        for k in self.param_names:
            if k in sg:
                rows.append(f"<tr><td><code>{k}</code></td><td>{self.params[k]:.6g}</td>"
                            f"<td>{sg[k]:.2g}</td>"
                            f"<td>{PARAM_UNITS.get(k, 'x-unit')}</td></tr>")
        self.w_result.value = ("<table style='font-size:12px'>" + "".join(rows) + "</table>"
                               f"<small>rms {self.result['rms_GHz']*1e3:.1f} MHz on "
                               f"{self.result['n_points']} points; uncertainties are the "
                               "diagonal of the covariance from the Jacobian and ignore "
                               "systematic line-extraction error</small>")

    # ---- drawing
    def _update_image(self, name):
        p = self.panels[name]
        s = self.maps.get(name)
        if s is None:
            return
        lo, hi = np.percentile(s, self.w_clim.value)
        p["im"].set_data(s); p["im"].set_clim(lo, hi); p["im"].set_cmap(self.w_cmap.value)
        p["fig"].canvas.draw_idle()

    def _draw_model(self):
        for sc in self.scans:
            p = self.panels[sc.name]
            for t in p["texts"]:
                t.remove()
            p["texts"] = []
            xs = np.linspace(sc.x[0], sc.x[-1], self.w_nmodel.value)
            try:
                br = self.model.branches(self.params, x_eff(self.params, sc.name, xs),
                                         f_window=(sc.f[0], sc.f[-1]))
            except Exception as e:
                self._log(f"model error: {e}"); return
            segs, cols = [], []
            for lab, (fr, w) in br.items():
                c = _lab_color(lab)
                for seg in _split_nan(xs, fr):
                    segs.append(seg); cols.append(c)
                if self.w_annot.value:
                    m = np.isfinite(fr) & (fr >= sc.f[0]) & (fr <= sc.f[-1])
                    if m.any():
                        i = np.flatnonzero(m)[-1]
                        p["texts"].append(p["ax"].text(xs[i], fr[i], " " + label_str(lab),
                                                       color=c, fontsize=7, va="center",
                                                       zorder=8))
            p["model"].set_segments(segs); p["model"].set_color(cols)
            p["ax"].set_xlim(sc.x[0], sc.x[-1]); p["ax"].set_ylim(sc.f[0], sc.f[-1])
            p["fig"].canvas.draw_idle()
        if self._ui_built:
            labs = [(label_str(l), l) for l in sorted(
                {t.label for t in self.traces if t.label}, key=label_str)]
            try:
                x9 = self.scans[0].x[::max(1, self.scans[0].x.size // 9)]
                br = self.model.branches(self.params,
                                         x_eff(self.params, self.scans[0].name, x9))
                labs = [(label_str(l), l) for l in sorted(br, key=label_str)]
            except Exception:
                pass
            cur = self.w_lab.value
            self.w_lab.options = [("(none)", None)] + labs
            if cur in [o[1] for o in self.w_lab.options]:
                self.w_lab.value = cur

    def _draw_traces(self):
        for sc in self.scans:
            p = self.panels[sc.name]
            segs, cols, lws, sty = [], [], [], []
            for t in self.traces:
                if t.scan != sc.name:
                    continue
                segs.append(np.column_stack([t.x, t.f]))
                cols.append(_lab_color(t.label) if t.label else (1, 1, 1, 0.9))
                lws.append(1.8 if t.include else 0.8)
                sty.append("solid" if t.include else (0, (1, 2)))
            p["traces"].set_segments(segs)
            if segs:
                p["traces"].set_color(cols); p["traces"].set_linewidth(lws)
                p["traces"].set_linestyle(sty)
            t = self.selected
            p["sel"].set_data((t.x, t.f) if (t is not None and t.scan == sc.name)
                              else ([], []))
            d = self._draw_pts[sc.name]
            p["clicks"].set_data(([q[0] for q in d], [q[1] for q in d]) if d else ([], []))
            p["fig"].canvas.draw_idle()

    def _refresh_traces(self):
        self._draw_traces()
        rows = []
        for t in sorted(self.traces, key=lambda t: (t.scan, t.tid)):
            cb = W.Checkbox(value=t.include, indent=False, layout=W.Layout(width="26px"))
            cb.observe(lambda c, t=t: (setattr(t, "include", c["new"]), self._draw_traces()),
                       "value")
            bs = W.Button(description="sel", layout=W.Layout(width="42px"))
            bs.on_click(lambda _, t=t: (setattr(self, "selected", t), self._draw_traces(),
                                        self._set_status()))
            bd = W.Button(description="x", layout=W.Layout(width="30px"))
            bd.on_click(lambda _, t=t: self._delete(t))
            star = "*" if t.manual_label else " "
            txt = (f"#{t.tid}{star} {t.scan} {len(t):4d}pt "
                   f"{label_str(t.label) if t.label else '-':>16s} "
                   f"conf {t.confidence:.1f} " if np.isfinite(t.confidence) else
                   f"#{t.tid}{star} {t.scan} {len(t):4d}pt "
                   f"{label_str(t.label) if t.label else '-':>16s} ")
            if np.isfinite(t.residual):
                txt += f"res {t.residual*1e3:.0f} MHz"
            rows.append(W.HBox([cb, bs, bd, W.HTML(f"<code>{txt}</code>")]))
        self.w_lines.children = tuple(rows)
        self._set_status()

    # ---- canvas events
    def _on_click(self, event, name):
        p = self.panels[name]
        if event.inaxes is not p["ax"] or event.button != 1:
            return
        tb = getattr(p["fig"].canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        mode = self.w_mode.value
        if mode == "draw":
            self._draw_pts[name].append((event.xdata, event.ydata)); self._draw_traces(); return
        if mode == "calibrate":
            self._calibrate_click(name, event.xdata); return
        t = self._nearest(name, event)
        if t is None:
            self._set_status("no line within 12 px of the click"); return
        if mode == "toggle":
            t.include = not t.include
            self._set_status(f"line #{t.tid} {'included' if t.include else 'excluded'}")
            self._draw_traces()
        elif mode == "label":
            self.selected = t
            self._apply_label()
        elif mode == "select":
            self.selected = t; self._draw_traces(); self._set_status()
        elif mode == "erase":
            self._delete(t)
        elif mode == "trim":
            keep = t.x >= event.xdata if abs(event.xdata - t.x[0]) > abs(event.xdata - t.x[-1]) \
                else t.x <= event.xdata
            if keep.sum() >= 3:
                t.x, t.f, t.w = t.x[keep], t.f[keep], t.w[keep]
                t.source = "manual"
                self._refresh_traces()
            else:
                self._set_status("refusing to trim below 3 points")

    def _nearest(self, name, event, max_px=12):
        ax = self.panels[name]["ax"]
        best, bd = None, max_px
        for t in self.traces:
            if t.scan != name or t.x.size == 0:
                continue
            xd, fd, _ = t.decimate(300)
            pts = ax.transData.transform(np.column_stack([xd, fd]))
            d = float(np.min(np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)))
            if d < bd:
                bd, best = d, t
        return best

    def _finish_draw(self):
        made = 0
        for sc in self.scans:
            pts = self._draw_pts[sc.name]
            if len(pts) < 2:
                continue
            self._ensure_maps()
            r = snap_to_ridge(self.maps[sc.name], sc.x, sc.f, [q[0] for q in pts],
                              [q[1] for q in pts], window_ghz=self.w_snapwin.value / 1e3,
                              median_smooth=self.w_medsm.value)
            self._draw_pts[sc.name] = []
            if r is None:
                continue
            x, f, w = r
            self.traces.append(Trace(scan=sc.name, x=x, f=f, w=w, tid=self._next_tid,
                                     source="manual"))
            self._next_tid += 1; made += 1
        self._log(f"added {made} manual line(s)" if made else
                  "need at least 2 clicked points in a panel")
        self._refresh_traces()

    def _cancel_draw(self):
        self._draw_pts = {s.name: [] for s in self.scans}
        self._draw_traces()

    def _delete(self, t):
        self.traces = [q for q in self.traces if q is not t]
        if self.selected is t:
            self.selected = None
        self._refresh_traces()

    def _apply_label(self):
        t = self.selected
        if t is None:
            self._set_status("select a line first: mode 'label', mode 'select', or the "
                             "'sel' button in the Lines tab"); return
        t.label = self.w_lab.value
        t.manual_label = t.label is not None
        t.confidence = np.inf if t.label else np.nan
        self._log(f"line #{t.tid} ({t.scan}) labelled "
                  + (f"{label_str(t.label)} by hand; automatic assignment will leave it "
                     "alone" if t.label else "(none)"))
        self._refresh_traces()

    def _unlock_labels(self):
        n = sum(1 for t in self.traces if t.manual_label)
        for t in self.traces:
            t.manual_label = False
        self._log(f"unlocked {n} hand-set label(s); they can now be reassigned")
        self._refresh_traces()

    def _set_all(self, v):
        for t in self.traces:
            t.include = v
        self._refresh_traces()

    def _del_unlabelled(self):
        self.traces = [t for t in self.traces if t.label is not None]
        self._refresh_traces()
