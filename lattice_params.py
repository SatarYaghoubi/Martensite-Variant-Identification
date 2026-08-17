#!/usr/bin/env python3
"""Back-calculate the B19' lattice parameters from the measured stretch tensors."""
import numpy as np, identify_martensite as m
np.set_printoptions(precision=4, suppress=True)

A0 = 2.9994                      # MEAM B2 reference
EXP = dict(a=2.889, b=4.120, c=4.622, alpha=90.0, beta=96.8, gamma=90.0)

def variants_with_rotations(U1, tol=1e-8):
    out = []
    for R in m.cubic_rotations():
        U = R @ U1 @ R.T
        if not any(np.allclose(U, V, atol=tol) for V, _ in out):
            out.append((U, R))
    return out

def cell_from_U(U, a0):
    """Apply U to the B2->B19' correspondence vectors and read off a,b,c,alpha,beta,gamma."""
    E = a0 * np.array([[1.,0.,0.],[0.,1.,-1.],[0.,1.,1.]])   # cols: a0[100], a0[011], a0[0-11]
    M = U @ E
    v = [M[:, k] for k in range(3)]
    n = [np.linalg.norm(x) for x in v]
    ang = lambda p, q: np.degrees(np.arccos(np.dot(v[p], v[q])/(n[p]*n[q])))
    return dict(a=n[0], b=n[1], c=n[2],
                alpha=ang(1,2), beta=ang(0,2), gamma=ang(0,1),
                vol=abs(np.linalg.det(M)))

U1 = m.build_U1(A0, 2.89, 4.12, 4.62, 96.8)
VR = variants_with_rotations(U1)
V  = [U for U, _ in VR]

# --- sanity check: the construction must return its own input ---
print("sanity check (U1 applied to E must reproduce a=2.89 b=4.12 c=4.62 beta=96.8):")
c = cell_from_U(U1, A0)
print(f"  a={c['a']:.4f} b={c['b']:.4f} c={c['c']:.4f} "
      f"alpha={c['alpha']:.2f} beta={c['beta']:.2f} gamma={c['gamma']:.2f}\n")

# --- measure on the detwinned file ---
ids, ty, pr, Hr = m.read_lammps_data('relaxed_NiTi_B2_meam.data')
_,  _,  pc, Hc = m.read_lammps_data('out.detwinned_martensite.data')
neigh = m.neighbour_list(pr, Hr, 3.5)
F = m.smooth_F(m.local_F(pr, Hr, pc, Hc, neigh), neigh, 1)
U = m.right_stretch(F)
lab, dist = m.classify(U, V, 0.10)

print("=" * 72)
print("MEASURED LATTICE PARAMETERS  (out.detwinned_martensite.data)")
print("=" * 72)

rows = []
for v in sorted(set(lab[lab > 0].tolist())):
    sel = lab == v
    Rv = VR[v-1][1]
    Uf = np.einsum('ji,njk,kl->nil', Rv, U[sel], Rv).mean(axis=0)   # Rv^T U Rv
    rows.append((f"variant {v}  ({sel.sum()} atoms)", cell_from_U(Uf, A0), Uf))

# all martensite together, each rotated into the variant-1 frame
Uall = []
for v in sorted(set(lab[lab > 0].tolist())):
    Rv = VR[v-1][1]
    Uall.append(np.einsum('ji,njk,kl->nil', Rv, U[lab == v], Rv))
Uall = np.concatenate(Uall).mean(axis=0)
rows.append((f"ALL martensite ({(lab>0).sum()} atoms)", cell_from_U(Uall, A0), Uall))

hdr = f"{'':<30}{'a (A)':>9}{'b (A)':>9}{'c (A)':>9}{'alpha':>9}{'beta':>9}{'gamma':>9}"
print(hdr); print("-" * len(hdr))
for name, c, _ in rows:
    print(f"{name:<30}{c['a']:>9.3f}{c['b']:>9.3f}{c['c']:>9.3f}"
          f"{c['alpha']:>9.2f}{c['beta']:>9.2f}{c['gamma']:>9.2f}")
print("-" * len(hdr))
print(f"{'EXPERIMENT (Kudoh 1985)':<30}{EXP['a']:>9.3f}{EXP['b']:>9.3f}{EXP['c']:>9.3f}"
      f"{EXP['alpha']:>9.2f}{EXP['beta']:>9.2f}{EXP['gamma']:>9.2f}")
print(f"{'THESIS report (twinned)':<30}{2.870:>9.3f}{4.181:>9.3f}{4.588:>9.3f}"
      f"{89.8:>9.2f}{96.0:>9.2f}{90.4:>9.2f}")

c = rows[-1][1]
print("\nerror vs experiment (all-martensite average):")
for k in ('a','b','c'):
    print(f"  {k}: {100*(c[k]-EXP[k])/EXP[k]:+.2f} %")
for k in ('alpha','beta','gamma'):
    print(f"  {k}: {c[k]-EXP[k]:+.2f} deg")

Uf = rows[-1][2]
ev = np.sort(np.linalg.eigvalsh(Uf))
print(f"\nmean U in the variant-1 frame:\n{Uf}")
print(f"det(U) = {np.linalg.det(Uf):.4f}   eigenvalues = {ev}   lambda2 = {ev[1]:.4f}")
Eexp = 2*(3.015**3)
print(f"cell volume = {c['vol']:.3f} A^3   experimental B19' cell = "
      f"{EXP['a']*EXP['b']*EXP['c']*np.sin(np.radians(EXP['beta'])):.3f} A^3")
