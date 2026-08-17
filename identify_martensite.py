#!/usr/bin/env python3
"""
identify_martensite.py
======================
Atom-by-atom identification of the B19' martensite variants (and residual B2
austenite) in a NiTi molecular dynamics simulation.

Method
------
1. Read a REFERENCE configuration (the relaxed B2 austenite) and a CURRENT
   configuration (e.g. the twinned martensite after cooling, or the detwinned
   structure after shear). Both must be LAMMPS data files with the same atom IDs.
2. For every atom, build the list of its 1st + 2nd B2 neighbours (cutoff ~3.5 A)
   in the reference configuration.
3. Compute the local deformation gradient F_i of every atom by least squares
   over its neighbour bond vectors (Falk & Langer style):
        F = (sum d d0^T) (sum d0 d0^T)^-1
   where d0 = reference bond vector, d = current bond vector (minimum image).
4. Polar-decompose F = R U  ->  local right stretch tensor U = sqrt(F^T F).
   U is expressed in the cubic austenite frame (the simulation box axes),
   i.e. exactly the transformation stretch tensor of the geometric theory.
5. Build the 12 theoretical variant matrices U_1 ... U_12 for the cubic ->
   monoclinic-I transformation (Bhattacharya 2003) from the lattice parameters
   (a0 | a, b, c, beta) and classify each atom to the closest matrix
   (Frobenius norm), including the identity matrix as "austenite".
6. Write:
     - a LAMMPS dump file with per-atom variant labels (open in OVITO and
       colour by the "variant" column to see the twins),
     - a text summary (variant fractions, mean U per variant, det(U),
       eigenvalues of the mean martensite U),
     - a PNG figure with slices through the box coloured by variant.

Usage
-----
    python3 identify_martensite.py  reference_B2.data  current.data \
        [--a0 3.019 --a 2.89 --b 4.12 --c 4.62 --beta 96.8] \
        [--cutoff 3.5] [--smooth 1] [--max-dist 0.08] \
        [--out-prefix variants]

Only numpy (and matplotlib for the optional figure) is required.
"""

import argparse
import sys
import numpy as np

# ----------------------------------------------------------------------------
# 1. LAMMPS data file reader (atomic style, orthogonal or triclinic box)
# ----------------------------------------------------------------------------

def read_lammps_data(path):
    """Return (ids, types, positions, H) from a LAMMPS data file.

    H is the 3x3 box matrix with LAMMPS convention:
        a = (lx, 0, 0), b = (xy, ly, 0), c = (xz, yz, lz)  (columns of H)
    """
    with open(path, "r") as fh:
        lines = [ln.strip() for ln in fh.read().replace("\r", "").split("\n")]

    natoms = None
    xlo = xhi = ylo = yhi = zlo = zhi = None
    xy = xz = yz = 0.0
    avec = bvec = cvec = None          # OVITO-style cell vectors
    atoms_start = None

    for i, ln in enumerate(lines):
        if ln.endswith("atoms"):
            natoms = int(ln.split()[0])
        elif "xlo xhi" in ln:
            xlo, xhi = map(float, ln.split()[:2])
        elif "ylo yhi" in ln:
            ylo, yhi = map(float, ln.split()[:2])
        elif "zlo zhi" in ln:
            zlo, zhi = map(float, ln.split()[:2])
        elif "xy xz yz" in ln:
            xy, xz, yz = map(float, ln.split()[:3])
        elif ln.endswith("avec"):
            avec = np.array(list(map(float, ln.split()[:3])))
        elif ln.endswith("bvec"):
            bvec = np.array(list(map(float, ln.split()[:3])))
        elif ln.endswith("cvec"):
            cvec = np.array(list(map(float, ln.split()[:3])))
        elif ln.startswith("Atoms"):
            atoms_start = i + 1
            break

    if natoms is None or atoms_start is None:
        raise ValueError(f"Could not parse header of {path}")

    # OVITO writes "avec / bvec / cvec" lines instead of "xlo xhi ..."; convert
    # them to the LAMMPS lx/ly/lz + xy/xz/yz convention.
    if xlo is None and avec is not None:
        xlo, ylo, zlo = 0.0, 0.0, 0.0
        xhi, yhi, zhi = float(avec[0]), float(bvec[1]), float(cvec[2])
        xy, xz, yz = float(bvec[0]), float(cvec[0]), float(cvec[1])

    ids = np.empty(natoms, dtype=int)
    types = np.empty(natoms, dtype=int)
    pos = np.empty((natoms, 3))
    count = 0
    for ln in lines[atoms_start:]:
        if not ln:
            continue
        parts = ln.split()
        if not parts[0].lstrip("-").isdigit():
            break  # next section (Velocities, ...)
        ids[count] = int(parts[0])
        types[count] = int(parts[1])
        pos[count] = [float(parts[2]), float(parts[3]), float(parts[4])]
        count += 1
        if count == natoms:
            break
    if count != natoms:
        raise ValueError(f"Expected {natoms} atoms in {path}, found {count}")

    lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
    H = np.array([[lx, xy, xz],
                  [0.0, ly, yz],
                  [0.0, 0.0, lz]])

    # sort by atom id so that the two files line up row-by-row
    order = np.argsort(ids)
    return ids[order], types[order], pos[order], H


def min_image(d, H, Hinv):
    """Minimum-image convention for a set of displacement vectors d (N,3)."""
    s = d @ Hinv.T
    s -= np.round(s)
    return s @ H.T


# ----------------------------------------------------------------------------
# 2. Theoretical variant matrices (cubic -> monoclinic I, Bhattacharya 2003)
# ----------------------------------------------------------------------------

def build_U1(a0, a, b, c, beta_deg):
    """Transformation stretch tensor of one variant, built directly from the
    B2 -> B19' lattice correspondence (a <- a0[100], b <- a0[011],
    c <- a0[0-11]).  This is equivalent to the closed-form (gamma, epsilon,
    alpha, delta) expressions of Bhattacharya (2003) but immune to typos:
    U = sqrt(F^T F) with F mapping the reference cell onto the monoclinic
    cell (unique axis b, angle beta between a and c)."""
    beta = np.radians(beta_deg)
    # reference lattice vectors as columns (austenite frame)
    E = a0 * np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, -1.0],
                       [0.0, 1.0, 1.0]])
    # deformed monoclinic cell in an arbitrary orthonormal frame
    M = np.array([[a, 0.0, c * np.cos(beta)],
                  [0.0, b, 0.0],
                  [0.0, 0.0, c * np.sin(beta)]])
    F = M @ np.linalg.inv(E)
    C = F.T @ F
    w, V = np.linalg.eigh(C)
    return V @ np.diag(np.sqrt(np.clip(w, 1e-12, None))) @ V.T


def cubic_rotations():
    """The 24 proper rotations of the cube (signed permutation matrices with
    determinant +1)."""
    mats = []
    from itertools import permutations, product
    for perm in permutations(range(3)):
        for signs in product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                R[row, col] = s
            if np.isclose(np.linalg.det(R), 1.0):
                mats.append(R)
    return mats


def build_variants(U1, tol=1e-8):
    """Orbit of U1 under the cubic point group -> the 12 distinct variants."""
    variants = []
    for R in cubic_rotations():
        U = R @ U1 @ R.T
        if not any(np.allclose(U, V, atol=tol) for V in variants):
            variants.append(U)
    return variants


# ----------------------------------------------------------------------------
# 3. Local deformation gradients and polar decomposition
# ----------------------------------------------------------------------------

def neighbour_list(pos, H, cutoff):
    """Brute-force neighbour list with minimum image (fine for <~2e4 atoms)."""
    Hinv = np.linalg.inv(H)
    n = len(pos)
    neigh = [[] for _ in range(n)]
    cut2 = cutoff * cutoff
    for i in range(n):
        d = min_image(pos[i + 1:] - pos[i], H, Hinv)
        r2 = np.einsum("ij,ij->i", d, d)
        for k in np.nonzero(r2 < cut2)[0]:
            j = i + 1 + k
            neigh[i].append(j)
            neigh[j].append(i)
    return neigh


def local_F(pos_ref, H_ref, pos_cur, H_cur, neigh):
    """Per-atom deformation gradient by least squares over neighbour bonds."""
    Hinv_ref = np.linalg.inv(H_ref)
    Hinv_cur = np.linalg.inv(H_cur)
    n = len(pos_ref)
    F = np.zeros((n, 3, 3))
    for i in range(n):
        js = np.asarray(neigh[i])
        d0 = min_image(pos_ref[js] - pos_ref[i], H_ref, Hinv_ref)   # (m,3)
        d1 = min_image(pos_cur[js] - pos_cur[i], H_cur, Hinv_cur)   # (m,3)
        A = d1.T @ d0          # sum d d0^T
        B = d0.T @ d0          # sum d0 d0^T
        F[i] = A @ np.linalg.inv(B)
    return F


def right_stretch(F):
    """U = sqrt(F^T F) for an array of deformation gradients (n,3,3)."""
    C = np.einsum("nij,nik->njk", F, F)          # F^T F
    w, V = np.linalg.eigh(C)                     # (n,3), (n,3,3)
    w = np.clip(w, 1e-12, None)
    return np.einsum("nij,nj,nkj->nik", V, np.sqrt(w), V)


def smooth_F(F, neigh, passes):
    """Average F with the neighbours to damp thermal noise (optional)."""
    for _ in range(passes):
        Fnew = F.copy()
        for i in range(len(F)):
            js = neigh[i]
            Fnew[i] = (F[i] + F[js].sum(axis=0)) / (1 + len(js))
        F = Fnew
    return F


# ----------------------------------------------------------------------------
# 4. Classification
# ----------------------------------------------------------------------------

def classify(U_atoms, variants, max_dist):
    """Assign every atom to austenite (0), variant 1..12, or unknown (-1)."""
    refs = [np.eye(3)] + list(variants)          # index 0 = austenite
    refs = np.array(refs)                        # (13,3,3)
    diff = U_atoms[:, None, :, :] - refs[None, :, :, :]
    dist = np.sqrt(np.einsum("nvij,nvij->nv", diff, diff))
    label = np.argmin(dist, axis=1)              # 0..12
    best = dist[np.arange(len(label)), label]
    label = np.where(best <= max_dist, label, -1)
    return label, best


# ----------------------------------------------------------------------------
# 5. Output helpers
# ----------------------------------------------------------------------------

def write_dump(path, ids, types, pos, H, label, dist, detU):
    """LAMMPS dump file readable by OVITO (colour-code the 'variant' column)."""
    lx, ly, lz = H[0, 0], H[1, 1], H[2, 2]
    xy, xz, yz = H[0, 1], H[0, 2], H[1, 2]
    triclinic = any(abs(v) > 1e-10 for v in (xy, xz, yz))
    with open(path, "w") as fh:
        fh.write("ITEM: TIMESTEP\n0\n")
        fh.write(f"ITEM: NUMBER OF ATOMS\n{len(ids)}\n")
        if triclinic:
            fh.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            fh.write(f"{min(0, xy, xz, xy + xz):.8f} {lx + max(0, xy, xz, xy + xz):.8f} {xy:.8f}\n")
            fh.write(f"{min(0, yz):.8f} {ly + max(0, yz):.8f} {xz:.8f}\n")
            fh.write(f"0.0 {lz:.8f} {yz:.8f}\n")
        else:
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            fh.write(f"0.0 {lx:.8f}\n0.0 {ly:.8f}\n0.0 {lz:.8f}\n")
        fh.write("ITEM: ATOMS id type x y z variant match_dist detU\n")
        for k in range(len(ids)):
            fh.write(f"{ids[k]} {types[k]} "
                     f"{pos[k, 0]:.6f} {pos[k, 1]:.6f} {pos[k, 2]:.6f} "
                     f"{label[k]} {dist[k]:.4f} {detU[k]:.4f}\n")


def slice_figure(path, pos, H, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # colours: grey = unknown, black = austenite, 12 distinct variant colours
    var_cols = plt.get_cmap("tab20").colors
    cmap = {-1: (0.7, 0.7, 0.7), 0: (0.0, 0.0, 0.0)}
    for v in range(1, 13):
        cmap[v] = var_cols[(v - 1) % len(var_cols)]
    colours = np.array([cmap[l] for l in label])

    axes_pairs = [(1, 2, 0, "x"), (0, 2, 1, "y"), (0, 1, 2, "z")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    L = np.array([H[0, 0], H[1, 1], H[2, 2]])
    for ax, (u, v, w, name) in zip(axs, axes_pairs):
        mid = pos[:, w].mean()
        thick = 2.2  # ~ one B2 lattice slab
        sel = np.abs(pos[:, w] - mid) < thick
        ax.scatter(pos[sel, u], pos[sel, v], c=colours[sel], s=28)
        ax.set_aspect("equal")
        ax.set_title(f"slice through {name}-mid  ({name} = {mid:.1f} +/- {thick} A)")
        ax.set_xlabel("xyz"[u] + " (A)")
        ax.set_ylabel("xyz"[v] + " (A)")
    # legend
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap[0], label="austenite (B2)")]
    for v in sorted(set(label)):
        if v > 0:
            handles.append(plt.Line2D([], [], marker="o", ls="", color=cmap[v],
                                      label=f"variant {v}"))
    if -1 in label:
        handles.append(plt.Line2D([], [], marker="o", ls="", color=cmap[-1], label="unknown"))
    fig.legend(handles=handles, loc="upper center", ncol=min(7, len(handles)),
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Identify B19' martensite variants "
                                            "in a NiTi MD simulation.")
    p.add_argument("reference", help="LAMMPS data file of the relaxed B2 austenite")
    p.add_argument("current", help="LAMMPS data file of the structure to analyse")
    p.add_argument("--a0", type=float, default=3.019, help="B2 lattice parameter (A)")
    p.add_argument("--a", type=float, default=2.89, help="B19' a (A)")
    p.add_argument("--b", type=float, default=4.12, help="B19' b (A)")
    p.add_argument("--c", type=float, default=4.62, help="B19' c (A)")
    p.add_argument("--beta", type=float, default=96.8, help="B19' monoclinic angle (deg)")
    p.add_argument("--cutoff", type=float, default=3.5,
                   help="neighbour cutoff in the REFERENCE config (A); 3.5 A "
                        "captures the 8 first + 6 second B2 neighbours")
    p.add_argument("--smooth", type=int, default=1,
                   help="number of neighbour-averaging passes on F (thermal-noise filter)")
    p.add_argument("--max-dist", type=float, default=0.10,
                   help="max Frobenius distance for a positive identification")
    p.add_argument("--out-prefix", default="variants", help="prefix of output files")
    p.add_argument("--no-plot", action="store_true", help="skip the PNG figure")
    args = p.parse_args()

    ids_r, types_r, pos_r, H_r = read_lammps_data(args.reference)
    ids_c, types_c, pos_c, H_c = read_lammps_data(args.current)
    if not np.array_equal(ids_r, ids_c):
        sys.exit("ERROR: the two files do not contain the same atom IDs.")
    print(f"Read {len(ids_r)} atoms.")
    print(f"Reference box diag: {H_r[0,0]:.3f} {H_r[1,1]:.3f} {H_r[2,2]:.3f} A")
    print(f"Current   box diag: {H_c[0,0]:.3f} {H_c[1,1]:.3f} {H_c[2,2]:.3f} A")

    # theoretical variants
    U1 = build_U1(args.a0, args.a, args.b, args.c, args.beta)
    variants = build_variants(U1)
    print(f"\nBuilt {len(variants)} distinct variant matrices from "
          f"a0={args.a0}, a={args.a}, b={args.b}, c={args.c}, beta={args.beta} deg")
    print("U1 =\n", np.array_str(U1, precision=4, suppress_small=True))
    print(f"det(U1) = {np.linalg.det(U1):.4f}   "
          f"eigenvalues = {np.round(np.sort(np.linalg.eigvalsh(U1)), 4)}")

    # local deformation gradients
    print("\nBuilding neighbour list in the reference configuration ...")
    neigh = neighbour_list(pos_r, H_r, args.cutoff)
    nn = np.array([len(x) for x in neigh])
    print(f"neighbours per atom: min {nn.min()}, mean {nn.mean():.1f}, max {nn.max()}")

    print("Computing local deformation gradients ...")
    F = local_F(pos_r, H_r, pos_c, H_c, neigh)
    if args.smooth > 0:
        F = smooth_F(F, neigh, args.smooth)
    U = right_stretch(F)
    detU = np.linalg.det(U)

    # classification
    label, dist = classify(U, variants, args.max_dist)

    # ---------------- report ----------------
    lines = []
    lines.append("=" * 66)
    lines.append("MARTENSITE VARIANT IDENTIFICATION - SUMMARY")
    lines.append("=" * 66)
    lines.append(f"reference : {args.reference}")
    lines.append(f"current   : {args.current}")
    lines.append(f"atoms     : {len(ids_r)}")
    lines.append("")
    lines.append(f"{'label':<16}{'atoms':>8}{'fraction':>12}{'<match dist>':>14}")
    lines.append("-" * 50)
    for lab in [-1, 0] + list(range(1, 13)):
        m = label == lab
        if m.sum() == 0:
            continue
        name = {-1: "unknown", 0: "austenite B2"}.get(lab, f"variant {lab}")
        lines.append(f"{name:<16}{m.sum():>8}{m.sum() / len(label):>12.3f}"
                     f"{dist[m].mean():>14.4f}")
    lines.append("")
    mart = label > 0
    lines.append(f"martensite fraction : {mart.mean():.3f}")
    lines.append(f"det(U) per atom     : mean {detU.mean():.4f}, "
                 f"std {detU.std():.4f}  (volume change "
                 f"{100 * (detU.mean() - 1):+.2f} %)")
    if mart.sum() > 0:
        # mean stretch tensor of the dominant variant -> lattice-level check
        counts = np.bincount(label[mart], minlength=13)
        vdom = int(np.argmax(counts))
        Um = U[label == vdom].mean(axis=0)
        ev = np.sort(np.linalg.eigvalsh(Um))
        lines.append("")
        lines.append(f"dominant variant    : {vdom}  ({counts[vdom]} atoms)")
        lines.append("mean U of dominant variant =")
        for row in Um:
            lines.append("    [" + "  ".join(f"{v:+.4f}" for v in row) + "]")
        lines.append(f"det(mean U)         : {np.linalg.det(Um):.4f}")
        lines.append(f"eigenvalues         : l1={ev[0]:.4f}  l2={ev[1]:.4f}  "
                     f"l3={ev[2]:.4f}   (ideal: l2 = 1)")
    report = "\n".join(lines)
    print("\n" + report)

    with open(args.out_prefix + "_summary.txt", "w") as fh:
        fh.write(report + "\n")
    write_dump(args.out_prefix + ".dump", ids_c, types_c, pos_c, H_c,
               label, dist, detU)
    print(f"\nwrote {args.out_prefix}.dump  (open in OVITO, colour by 'variant')")
    print(f"wrote {args.out_prefix}_summary.txt")
    if not args.no_plot:
        slice_figure(args.out_prefix + "_slices.png", pos_c, H_c, label)
        print(f"wrote {args.out_prefix}_slices.png")


if __name__ == "__main__":
    main()
