# pymd Educational Examples

A series of runnable examples demonstrating molecular dynamics concepts.

## Quick Start

```bash
# Simple 2-atom tests (fast, < 5 seconds)
python examples/quick_test.py
python examples/01_harmonic_oscillator.py
python examples/02_diatomic_morse.py

# Larger systems with 32 atoms (slower, up to ~1 minute)
python examples/03_thermostat_comparison.py
python examples/04_velocity_distribution.py
python examples/05_melting_transition.py
python examples/06_equipartition.py
```

## Example Overview

| # | Example | Atoms | Concept | Time |
|---|---------|-------|---------|------|
| - | `quick_test.py` | 2 | Basic LJ NVE test | < 5s |
| 1 | `01_harmonic_oscillator.py` | 2 | Harmonic potential, energy conservation | < 5s |
| 2 | `02_diatomic_morse.py` | 2 | Morse potential, anharmonic vibration | < 5s |
| 3 | `03_thermostat_comparison.py` | 32 | NVE vs NVT ensemble | ~30s |
| 4 | `04_velocity_distribution.py` | 32 | Maxwell-Boltzmann statistics | ~30s |
| 5 | `05_melting_transition.py` | 32 | Lindemann melting criterion | ~30s |
| 6 | `06_equipartition.py` | 32 | Equipartition theorem | ~30s |

## Concepts Covered

### 1. Harmonic Oscillator
- Custom potential implementation
- Periodic oscillation
- Energy conservation in NVE

### 2. Diatomic Morse Molecule  
- Anharmonic potential
- Molecular vibration
- Bond length dynamics

### 3. Thermostat Comparison
- NVE (microcanonical) vs NVT (canonical)
- Temperature control
- Berendsen thermostat

### 4. Velocity Distribution
- Maxwell-Boltzmann statistics
- Velocity component analysis
- Speed distribution

### 5. Melting Transition
- Structure vs temperature
- Lindemann criterion (RMSD > 10% of lattice constant)
- Solid-liquid transition

### 6. Equipartition Theorem
- Energy distribution among DOFs
- <KE_x> = <KE_y> = <KE_z> = (1/2)*N*kB*T
- Statistical mechanics verification

## Performance Notes

Examples with 32 atoms use the numerical differentiation backend:
- Each force calculation requires 6N energy evaluations
- For 32 atoms: 192 energy calls per step
- Total time depends on number of steps

For faster execution:
- Reduce `steps` parameter in examples
- Use 2-atom examples for quick tests
