# Martensite-Variant-Identification
Per-atom identification of martensite variants in MD simulations. Computes local deformation gradients from LAMMPS data files, classifies atoms by transformation stretch tensor (B2→B19' NiTi built in, extensible to other cubic-parent transformations), and outputs OVITO-ready dumps.
# Martensite Variant Identification

Per-atom identification of martensite variants (and residual austenite) in molecular dynamics simulations.

Given a **reference** configuration (the relaxed parent phase) and a **current** configuration (e.g. twinned martensite after cooling, or a detwinned structure after shear), the code computes each atom's local transformation stretch tensor and assigns it to one of the crystallographic variants of the transformation. Input and output are LAMMPS data / dump files, so results plug directly into OVITO for visualization.

## Scope

- **Works out of the box**: the cubic → monoclinic-I transformation (B2 → B19'), i.e. NiTi and NiTi-based shape memory alloys — 12 variants, lattice parameters set from the command line.
- **Works with a small edit** to `build_U1()` (replace the lattice correspondence): any other transformation from a *cubic* parent, e.g. FCC → BCT in Fe-C steel via the Bain correspondence (3 variants), B2 → B19 orthorhombic in TiNiCu/AuCd (6 variants), or the NiTi R-phase (4 variants). The variant orbit and the classifier need no changes.
- **Needs more work**: transformations from a non-cubic parent (the parent point group in `cubic_rotations()` would have to be replaced).
- **Assumptions**: the transformation is displacive — both files must contain the same atoms with the same IDs (for interstitial-containing systems like Fe-C, analyse the host sublattice only) — and the neighbour cutoff must capture the first two coordination shells of *your* parent structure.

## Method

1. Read the reference and current LAMMPS data files (both must contain the same atom IDs).
2. Build every atom's 1st + 2nd neighbour shell in the reference configuration (cutoff ≈ 3.5 Å for B2 NiTi).
3. Compute the local deformation gradient `F` of every atom by least squares over its neighbour bond vectors (Falk & Langer style):
   `F = (Σ d d0ᵀ)(Σ d0 d0ᵀ)⁻¹`, where `d0` = reference bond vector and `d` = current bond vector (minimum image).
4. Polar-decompose `F = R U` → local right stretch tensor `U = sqrt(FᵀF)`, expressed in the parent (austenite) frame — i.e. exactly the transformation stretch tensor of the geometric theory of martensite.
5. Build the theoretical variant matrices `U1 … Un` as the orbit of one stretch tensor under the parent point group (Bhattacharya 2003). For B2 → B19' the built-in lattice correspondence is `a ← a0[100]`, `b ← a0[011]`, `c ← a0[0-11]`, giving 12 variants.
6. Classify each atom to the closest matrix (Frobenius norm), with the identity matrix included as "austenite".
   Labels: `0` = austenite, `1–n` = martensite variants, `-1` = unknown.

## Repository contents

| File | Purpose |
|---|---|
| `identify_martensite.py` | Main command-line tool: per-atom variant identification; writes an OVITO-readable dump, a text summary, and a slice figure |
| `lattice_params.py` | Back-calculates the martensite lattice parameters (a, b, c, α, β, γ) realized in a simulation from the measured per-atom stretch tensors, and compares them with experiment (NiTi example, Kudoh 1985) |
| `verify_report.py` | Verification suite: re-derives the variant matrices for different `a0`, classifier margins, twin-boundary averaging artifacts, twin relations between variant pairs, and variant orientations |

## Requirements

Python ≥ 3.8 with `numpy` (and `matplotlib` for the optional figure):

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python3 identify_martensite.py reference_austenite.data current.data \
    [--a0 3.019 --a 2.89 --b 4.12 --c 4.62 --beta 96.8] \
    [--cutoff 3.5] [--smooth 1] [--max-dist 0.10] \
    [--out-prefix variants] [--no-plot]
```

Example (NiTi, MEAM potential):

```bash
python3 identify_martensite.py relaxed_NiTi_B2_meam.data out.detwinned_martensite.data \
    --a0 2.9994 --out-prefix detwinned
```

| Option | Default | Meaning |
|---|---|---|
| `reference` | — | LAMMPS data file of the relaxed parent phase (B2 austenite) |
| `current` | — | LAMMPS data file of the structure to analyse |
| `--a0` | 3.019 | Parent (B2) lattice parameter (Å) |
| `--a` | 2.89 | B19' a (Å) |
| `--b` | 4.12 | B19' b (Å) |
| `--c` | 4.62 | B19' c (Å) |
| `--beta` | 96.8 | B19' monoclinic angle (deg) |
| `--cutoff` | 3.5 | Neighbour cutoff in the reference configuration (Å); 3.5 Å captures the 8 first + 6 second B2 neighbours |
| `--smooth` | 1 | Neighbour-averaging passes on `F` (thermal-noise filter) |
| `--max-dist` | 0.10 | Maximum Frobenius distance for a positive identification |
| `--out-prefix` | `variants` | Prefix of the output files |
| `--no-plot` | off | Skip the PNG figure |

Set the lattice parameters to values consistent with your interatomic potential, not necessarily the experimental ones: e.g. the B2 lattice constant of the NiTi MEAM potential used in this work is 2.9994 Å, while the experimental value is 3.015 Å. For noisy microstructures (heavy plastic accommodation, lath martensite), increase `--smooth` and loosen `--max-dist`.

## Outputs

- `<prefix>.dump` — LAMMPS dump file with per-atom columns `variant`, `match_dist`, `detU`. Open it in OVITO and colour by the `variant` column to see the twin microstructure.
- `<prefix>_summary.txt` — variant fractions, mean `U` of the dominant variant, `det(U)` (volume change), and its eigenvalues (compatibility check: middle eigenvalue λ₂ ≈ 1).
- `<prefix>_slices.png` — three orthogonal slices through the box, coloured by variant.

## Input files

- LAMMPS data files, `atomic` style. Orthogonal and triclinic boxes are supported, including OVITO-exported files that use `avec/bvec/cvec` lines.
- The reference and current files must contain the same atoms; they are matched row-by-row after sorting by atom ID.
- The neighbour list is brute-force O(N²), which is fine up to ~2×10⁴ atoms.

## Helper scripts

`lattice_params.py` and `verify_report.py` are NiTi-specific examples. They import `identify_martensite.py` as a module and expect `relaxed_NiTi_B2_meam.data` and `out.detwinned_martensite.data` in the working directory — edit the file names (and the reference constants at the top of `lattice_params.py`) to match your own runs.

```bash
python3 lattice_params.py
python3 verify_report.py
```

## Extending to other transformations

Only `build_U1()` encodes the transformation. It constructs `U = sqrt(FᵀF)` from `F = M E⁻¹`, where the columns of `E` are the parent cell vectors selected by the lattice correspondence and the columns of `M` are the corresponding martensite cell vectors. To add, for example, the FCC → BCT (Bain) transformation of steel, replace it with the Bain stretch `U1 = diag(√2·a/a0, √2·a/a0, c/a0)`; `build_variants()` will then automatically produce the 3 Bain variants. Remember to also retune `--cutoff` to the first two shells of the new parent structure.

## References

- K. Bhattacharya, *Microstructure of Martensite: Why It Forms and How It Gives Rise to the Shape-Memory Effect*, Oxford University Press (2003) — variant theory and transformation stretch tensors.
- M. L. Falk and J. S. Langer, Phys. Rev. E **57**, 7192 (1998) — least-squares local deformation gradient.
- Y. Kudoh, M. Tokonami, S. Miyazaki and K. Otsuka, Acta Metall. **33**, 2049 (1985) — experimental B19' lattice parameters used for the NiTi comparison.

## License

Released under the MIT License — see [LICENSE](LICENSE).
