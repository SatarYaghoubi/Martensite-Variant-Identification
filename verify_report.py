#!/usr/bin/env python3
"""Re-derives every numerical claim used in report_review.md."""
import numpy as np, identify_martensite as m
np.set_printoptions(precision=4, suppress=True)
I = np.eye(3)

print("== A2: variant matrices vs choice of a0 ==")
for a0, tag in [(3.015,"experimental"), (3.019,"code default"), (2.9994,"MEAM")]:
    U = m.build_U1(a0, 2.889 if a0==3.015 else 2.89, 4.12,
                   4.622 if a0==3.015 else 4.62, 96.8)
    ev = np.sort(np.linalg.eigvalsh(U))
    print(f" a0={a0} ({tag}): gamma={U[0,0]:.4f} eps={U[0,1]:.4f} alpha={U[1,1]:.4f} "
          f"delta={-U[1,2]:.4f} det={np.linalg.det(U):.4f} lam2={ev[1]:.4f}")

U1 = m.build_U1(2.9994, 2.89, 4.12, 4.62, 96.8); V = m.build_variants(U1)

print("\n== A5: classifier margins ==")
D = np.array([[np.linalg.norm(V[i]-V[j]) for j in range(12)] for i in range(12)])
print(f" ||U_i - I||   = {np.linalg.norm(V[0]-I):.4f}")
print(f" min ||U_i-U_j|| = {D[~np.eye(12,dtype=bool)].min():.4f}  -> decision boundary at "
      f"{D[~np.eye(12,dtype=bool)].min()/2:.4f}")

print("\n== A3: twin-boundary averaging artifact ==")
for i, j in [(9,10), (11,12)]:
    Uavg = m.right_stretch((0.5*(V[i-1]+V[j-1]))[None])[0]
    print(f" pair {i}-{j}: dist to I = {np.linalg.norm(Uavg-I):.4f}, "
          f"to nearest variant = {min(np.linalg.norm(Uavg-U) for U in V):.4f}")

print("\n== B1: twin relations ==")
rots = m.cubic_rotations()
for i, j in [(9,10),(11,12),(9,11),(9,12),(10,11),(10,12)]:
    ax = []
    for R in rots:
        if np.allclose(R@V[j-1]@R.T, V[i-1], atol=1e-6) and np.isclose(np.trace(R), -1):
            w, Q = np.linalg.eigh(R); e = Q[:, np.argmax(w)]
            ax.append(np.round(e/np.max(abs(e)), 3))
    print(f" {i}-{j}: {'compound' if len(ax)>1 else 'Type I/II'}  axes={ax}")

print("\n== D: variant short-axis orientation ==")
for i, U in enumerate(V, 1):
    w, Q = np.linalg.eigh(U); v = Q[:, 0]; v = v*np.sign(v[np.argmax(abs(v))])
    print(f" variant {i:>2}: short axis {np.round(v,3)}")

print("\n== E: run on the detwinned file ==")
ids, ty, pr, Hr = m.read_lammps_data('relaxed_NiTi_B2_meam.data')
_, _, pc, Hc = m.read_lammps_data('out.detwinned_martensite.data')
neigh = m.neighbour_list(pr, Hr, 3.5)
nn = np.array([len(x) for x in neigh])
print(f" neighbours per atom: min {nn.min()} mean {nn.mean():.1f} max {nn.max()}")
F0 = m.local_F(pr, Hr, pc, Hc, neigh)
for sm in (0, 1, 2):
    F = m.smooth_F(F0.copy(), neigh, sm) if sm else F0
    U = m.right_stretch(F)
    for md in (0.10, 0.30):
        lab, d = m.classify(U, V, md)
        print(f" smooth={sm} max-dist={md}: "
              f"{ {k:int((lab==k).sum()) for k in sorted(set(lab.tolist()))} }  "
              f"mean detU={np.linalg.det(U).mean():.4f}")
print(f" box volume ratio det(H_cur)/det(H_ref) = {np.linalg.det(Hc)/np.linalg.det(Hr):.5f}")
