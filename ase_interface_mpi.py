import numpy as np
import ase.units as unit
from scme_f2py import scme
import ase.parallel as mpi

from ase.calculators.calculator import Calculator

class SCME_PS(Calculator):

    def __init__(self, atoms, 
                 numerical=False, 
                 repeat=False, 
                 fix_oxygen=False,
                 parallel=False):
        self.name = 'FortranInterfaceToSCMEPS'
        self.energy = None
        self.forces = None

        self.parameters = {}

        self.numatoms = len(atoms)
        self.oidx = self.numatoms * 2 // 3
        self.molnum = self.numatoms // 3

        self.numerical = numerical
        self.repeat = repeat
        self.fix_oxygen = fix_oxygen

        self.world = mpi.world

        self.initial = True
        self.parallel = parallel

        if self.parallel:
            assert self.world.size == 1 or len(atoms) % self.world.size == 0


    def ase_to_scme(self, atoms):
        # Reindex in SCME order (HHHH..OO..)
        oxygens = atoms[[atom.index for atom in atoms if atom.symbol == 'O']]
        hydrogens = atoms[[atom.index for atom in atoms if atom.symbol == 'H']]
        scmeatoms = hydrogens + oxygens
        # Change coordinate system to cell-centered (SCME-Style)
        asecell = atoms.get_cell()
        c_mid = asecell.diagonal() * 0.5
        coords_asecell = scmeatoms.get_positions()
        coords_scmecell = coords_asecell - c_mid
        scmeatoms.set_positions(coords_scmecell)

        return scmeatoms

    def scme_to_ase(self, atoms):
        # Reindex in ASE order (OHHOHH..)
        # Not used, since the original atoms
        # object is never updated with SCME stuff
        mollist = []
        for i in range(self.molnum):
            mollist.append(i + self.oidx)
            mollist.append(i + i)
            mollist.append(i + i + 1)

        ase_atoms = atoms[[mollist]]
        return ase_atoms

    def f_scme_to_ase(self, f):
        # Convert force array back to ase coordinates
        mollist = []
        aseforces = np.zeros([self.numatoms, 3])
        for i in range(self.molnum):
            mollist.append(i + self.oidx)
            mollist.append(i + i)
            mollist.append(i + i + 1)

        for i in range(self.numatoms):
            aseforces[i, 0] = f[mollist[i], 0]
            aseforces[i, 1] = f[mollist[i], 1]
            aseforces[i, 2] = f[mollist[i], 2]
        return aseforces

    def calculate(self, atoms):
        self.positions = atoms.get_positions()
        constraints = atoms.constraints
        atoms.constraints = []

        pos = self.positions[:]
        scmepos = pos[:]
        cell = atoms.get_cell()

        numatoms = self.numatoms
        nummols = self.molnum

        # Atomwise MIC PBCs needed for SCME
        n = np.zeros(np.shape(scmepos))
        c_mid = cell.diagonal() * 0.5
        n = np.rint((c_mid - pos) / cell.diagonal())
        scmepos += n * cell.diagonal()

        scme_atoms = atoms[:]
        scme_atoms.set_positions(scmepos)

        # Convert to scme coordinates
        scme_atoms = self.ase_to_scme(scme_atoms)
        scme_coords = scme_atoms.get_positions()
        scme_coords = scme_coords.transpose()

        ff, epot = scme.scme_calculate(scme_coords, cell.diagonal())

        # Reshape and convert SCME forces to ASE forces
        f = np.reshape(ff, [numatoms, 3])
        aseforces = self.f_scme_to_ase(f)

        self.atoms = atoms
        self.atoms.constraints = constraints
        self.forces = aseforces
        self.energy = epot

    def get_potential_energy(self, atoms, force_consistent=False):
        self.update(atoms)
        return self.energy

    def get_forces(self, atoms):

        self.update(atoms)
        if self.numerical:
            self.numerical_forces(atoms)

        return self.forces

    def get_stress(self, atoms):
        return self.calculate_numerical_stress(atoms)

    def update(self, atoms):
        if self.energy is None:
            self.calculate(atoms)
        elif (self.positions != atoms.get_positions()).any():
            self.initial = True
            self.calculate(atoms)

    def numerical_forces(self, atoms, d=0.00001):
        #
        scme_atoms = atoms[:]
        scme_atoms.constraints = []

        # Hold on to unperturbed position
        p0 = scme_atoms.get_positions()
        F = np.zeros_like(p0)

        # Repeat?
        if self.repeat: # slice out
            rep = self.repeat[0]*self.repeat[1]*self.repeat[2]
            l_rep = int(len(atoms) / rep)
            slice_atoms = scme_atoms[:l_rep]
        else:
            slice_atoms = scme_atoms

        # Given the slice, need to slice it further into 
        # individual slices per processor.
        if self.parallel:
            size = self.world.size
            rank = self.world.rank

            part = len(slice_atoms) // size
            this_slice = slice_atoms[rank * part:(rank + 1) * part]
            idx = part * rank
        else:
            idx = 0
            this_slice = slice_atoms


        for i, atom in enumerate(this_slice):
            if self.fix_oxygen:
                if atom.symbol == 'O':
                    continue
            for j in [0,1,2]:
                p = p0.copy()
                p[i + idx, j] += d

                scme_atoms.set_positions(p, apply_constraint=False)
                self.get_potential_energy(scme_atoms)

                eplus = self.energy

                p[i + idx, j] -= 2*d

                scme_atoms.set_positions(p, apply_constraint=False)
                self.get_potential_energy(scme_atoms)

                eminus = self.energy

                F[i + idx, j] = (eminus - eplus) / (2 * d)

        # Collect results for each slice
        if self.parallel:
            self.collect_arrays(F, part)

        if self.repeat: # Fill in forces which should be available to all processors...
            # Does this break    
            for r in range(1,rep):
                F[int(l_rep * r):int(l_rep * (r + 1)),:] = F[:l_rep,:]

        self.forces = F

    def collect_arrays(self, F, part):
        size = self.world.size
        for i in range(size):
            if self.world.rank == i:
                for rank in range(size):
                    if rank != i:
                        self.world.send(F[i * part:(i + 1) * part, :], rank)
            else:
                self.world.receive(F[i * part:(i + 1) * part, :], i)
        
