from pymatgen.core.periodic_table import Element

elements = ["Ag", "Al", "B", "C", "Ca", "Co", "Cr", "Cu", "Fe", "Ga", "Ge", "Hf", "Li", "Mg", "Mn", "Mo", "N", "Nb", "Nd", "Ni", "Pd", "Re", "Sc", "Si", "Sn", "Ta", "Ti", "V", "W", "Y", "Zn", "Zr"]

masses = []
volumes = []

for elem in elements:
    e = Element(elem)
    masses.append(e.atomic_mass)
    volumes.append(e.molar_volume)  # Estimating volume using atomic radius (approximation)

print("Masses:", masses)
print("Volumes:", volumes)

