
"""Operators that work on slabs.
Allowed compositions are respected.
Identical indexing of the slabs are assumed for the cut-splice operator."""
from ase.build import fcc111, fcc110, fcc100, hcp10m10, bcc100, bcc110, bcc111, hcp0001
import random
import numpy as np
import copy
from ase.ga import get_neighbor_list
from ase.ga.utilities import  get_neighborlist, get_rdf
from utils.GeneralUtils import SurfaceAtomFinder


''' Individual functions'''

def get_chemical_symbols_array(atoms):
    return np.array(atoms.get_chemical_symbols(), dtype='U2')

def get_nndist(atoms, rmax, distance_matrix):
    
    """Returns an estimate of the nearest neighbor bond distance
    in the supplied atoms object given the supplied distance_matrix.

    The estimate comes from the first peak in the radial distribution
    function.
    """
    rmax = rmax  # No bonds longer than half of cell angstrom expected
    nbins = 200
    rdf, dists = get_rdf(atoms, rmax, nbins, distance_matrix)
    return dists[np.argmax(rdf)]


def get_nnmat(atoms, rmax, mic=False):
    """Calculate the nearest neighbor matrix as specified in
    S. Lysgaard et al., Top. Catal., 2014, 57 (1-4), pp 33-39

    Returns an array of average numbers of nearest neighbors
    the order is determined by self.elements.
    Example: self.elements = ["Cu", "Ni"]
    get_nnmat returns a single list [Cu-Cu bonds/N(Cu),
    Cu-Ni bonds/N(Cu), Ni-Cu bonds/N(Ni), Ni-Ni bonds/N(Ni)]
    where N(element) is the number of atoms of the type element
    in the atoms object.

    The distance matrix can be quite costly to calculate every
    time nnmat is required (and disk intensive if saved), thus
    it makes sense to calculate nnmat along with e.g. the
    potential energy and save it in atoms.info['data']['nnmat'].
    """
    if 'data' in atoms.info and 'nnmat' in atoms.info['data']:
        return atoms.info['data']['nnmat']
    elements = sorted(set(atoms.get_chemical_symbols()))
    nnmat = np.zeros((len(elements), len(elements)))
    # dm = get_distance_matrix(atoms)
    dm = atoms.get_all_distances(mic=mic)
    nndist = get_nndist(atoms, rmax, dm) + 0.2
    for i in range(len(atoms)):
        row = [j for j in range(len(elements))
               if atoms[i].symbol == elements[j]][0]
        neighbors = [j for j in range(len(dm[i])) if dm[i][j] < nndist]
        for n in neighbors:
            column = [j for j in range(len(elements))
                      if atoms[n].symbol == elements[j]][0]
            nnmat[row][column] += 1
    # divide by the number of that type of atoms in the structure
    for i, el in enumerate(elements):
        nnmat[i] /= len([j for j in range(len(atoms))
                         if atoms[int(j)].symbol == el])
    # makes a single list out of a list of lists
    nnlist = np.reshape(nnmat, (len(nnmat)**2))
    return nnlist


def get_connections_index(atoms, max_conn=5, no_count_types=None):
    """This method returns a dictionary where each key value are a
    specific number of neighbors and list of atoms indices with
    that amount of neighbors respectively. The method utilizes the
    neighbor list and hence inherit the restrictions for
    neighbors. Option added to remove connections between
    defined atom types.

    Parameters
    ----------

    atoms : Atoms object
        The connections will be counted using this supplied Atoms object

    max_conn : int
        Any atom with more connections than this will be counted as
        having max_conn connections.
        Default 5

    no_count_types : list or None
        List of atomic numbers that should be excluded in the count.
        Default None (meaning all atoms count).
    """
    conn = get_neighbor_list(atoms)

    if conn is None:
        conn = get_neighborlist(atoms)

    if no_count_types is None:
        no_count_types = []

    conn_index = {}
    for i in range(len(atoms)):
        if atoms[i].number not in no_count_types:
            cconn = min(len(conn[i]), max_conn - 1)
            if cconn not in conn_index:
                conn_index[cconn] = []
            conn_index[cconn].append(i)

    return conn_index


def permute1(atoms, indices):
    i1 = indices[0]
    i2 = indices[1]
    sym1 = atoms[i1].symbol
    sym2 = atoms[i2].symbol
    if sym1 != sym2:
        atoms[i1].symbol = sym2
        atoms[i2].symbol = sym1
        print(f'permute result {i1} = {sym2} | {i2} = {sym1}')
    else:
        print(f'non sense to "permute" the atoms have the same element')
    
def permute2(atoms1, atoms2, indices):
    i1 = indices[0]
    i2 = indices[1]
    sym1 = atoms1[i1].symbol
    sym2 = atoms2[i2].symbol
    if sym1 != sym2:
        atoms1[i1].symbol = sym2
        atoms2[i2].symbol = sym1
        print(f'permute result {i1} = {sym2} | {i2} = {sym1}')
    else:
        print(f'non sense to "permute" the atoms have the same element')
        
def replace_element(atoms, index, element_old, element_new):
    syms = atoms.get_chemical_symbols()
    # extant element checling
    if syms[index] == element_old:
        syms[index] = str(element_new)
        atoms.set_chemical_symbols(syms)
        print(f'atom {index} | {element_old} -> {element_new}')
    else:
        print(f'non sense to "mutate" the atom have the same element {syms[index]} |{element_old}, {element_new}')

def dummy_func(*args):
    return


def get_slab_strings(atoms):
    
    try:
        atoms_fomula = atoms.info['key_value_pairs']['atoms_fomula']
    except:
        atoms_fomula = None
        print('atoms_fomula = None')
    
    try:
        atoms_string = atoms.info['key_value_pairs']['atoms_string']
    except:
        atoms_string = None
        print('atoms_string = None')
    
    try:
        atoms_nnmat_string = atoms.info['key_value_pairs']['nnmat_string']
    except:
        atoms_nnmat_string = None
        print('atoms_nnmat_string = None')
        
    
    return [atoms_fomula, atoms_string, atoms_nnmat_string]


    
'''The basic operators to support advanced operator'''
class SlabOperator():
    
    def __init__(self, rmax = None, element_pools=None, rng=None, NelementLimit = None):
        
        self.descriptor = 'Initialization'
        self.element_pools = element_pools
        self.rmax = rmax
        self.dcf = dummy_func
        self.NelementLimit = NelementLimit
        
    def get_all_element_mutations(self, a):
        """Get all possible mutations for the supplied atoms object given
        the element pools."""
        muts = []
        syms = a.get_chemical_symbols()
        elementset = set(syms)
        NnewElement = self.NelementLimit - len(elementset)
        subElementpool = set(self.element_pools) - elementset 
        subElementpool = random.sample(subElementpool, NnewElement)
        subElementpool = list(elementset) + subElementpool  
        print(f'Mutation Element Pool : {subElementpool}')
        for c, sym in enumerate(syms):
            for element in subElementpool:
                if element != sym:
                    muts.extend([(c, sym, element)])
        return muts
    
    
    def get_all_surface_element_mutations(self, a):
        """Get all possible mutations for the supplied atoms object given
        the element pools."""
        muts = []
        surfatomslist = SurfaceAtomFinder(a)
        surfatomsym = a.get_chemical_symbols()
        surfatomsym = [surfatomsym[i] for i in surfatomslist] 
        
        syms = a.get_chemical_symbols()
        elementset = set(syms)
        NnewElement = self.NelementLimit - len(elementset)
        subElementpool = set(self.element_pools) - elementset 
        subElementpool = random.sample(subElementpool, NnewElement)
        subElementpool = list(elementset) + subElementpool  
        print(f'Surface Mutation Element Pool : {subElementpool}')
        for ind, sym in zip(surfatomslist, surfatomsym):
            for element in subElementpool:
                if element != sym:
                    muts.extend([(ind, sym, element)])
        return muts
    
    
    def get_all_element_permutations(self, a):
        """Get all possible permutations for the supplied atoms object given
        the element pools."""
        permutationIndices = []
        syms = list(a.get_chemical_symbols())
        for i,symi in enumerate(syms):
            for j in range(i + 1, len(syms)):    
                if i != j and syms[i] != syms[j]:
                    permutationIndices.append([i, j])

        return permutationIndices
    
    
    def get_all_element_permutations_2slabs(self, atoms1, atoms2):
        """Get all possible permutations for the supplied atoms object given
        the element pools."""
        
        syms1 = list(atoms1.get_chemical_symbols())
        syms2 = list(atoms2.get_chemical_symbols())
        
        permutationIndices = []

        for i in range(len(syms1)):
            for j in range(len(syms2)):    
                if syms1[i] != syms2[j]:
                    permutationIndices.append([i, j])

        return permutationIndices
    
    def get_nnmat_string(self,atoms, decimals=2, mic=False):
        nnmat = get_nnmat(atoms, self.rmax, mic=mic)
        s = '-'.join(['{1:2.{0}f}'.format(decimals, i)
                      for i in nnmat])
        if len(nnmat) == 1:
            return s + '-'
        return s

    def initialize_individual(self, indi,  facet = ''):
        """Initializes a new individual that inherits some parameters
        from the parent, and initializes the info dictionary.
        If the new individual already has more structure it can be
        supplied in the parameter indi."""
        indi = copy.deepcopy(indi)
        # key_value_pairs for numbers and strings
        try:
            indi.info['data']
        except:
            indi.info['data'] = {}
            
        try:
            indi.info['key_value_pairs']
        except:
            indi.info['key_value_pairs'] = {}
        
        try:
            indi.info['key_value_pairs']['OperatorGenealogy']
        except:
            indi.info['key_value_pairs']['OperatorGenealogy'] = 'Initialization'
            
        try:
            indi.info['key_value_pairs']['Genealogy']
        except:
            indi.info['key_value_pairs']['Genealogy'] = 'Initialization'
            
        '''adding info for duplicated checking'''
        # chemical symbols list for checking 

        atoms_string = ''.join(indi.get_chemical_symbols())
        indi.info['key_value_pairs']['atoms_string'] = atoms_string

        nnmat_string = self.get_nnmat_string(indi, 2, True)
        indi.info['key_value_pairs']['nnmat_string'] = nnmat_string
            
        atoms_fomula = ''.join(indi.get_chemical_formula())
        indi.info['key_value_pairs']['atoms_fomula'] = atoms_fomula
        
        # assign info and genetic operator
        indi.info['key_value_pairs']['LastOperator'] = self.descriptor
        if facet != '':
            indi.info['key_value_pairs']['facet'] = facet

        return indi

'''Advanced Operators'''
class RandomElementMutation(SlabOperator):
    
    '''Mutate elements in one slab
       Accomodate multi slabs in a list 
    '''
    def __init__(self, element_pools = None, NelementLimit = None, rmax = 4.25, rng=np.random):
        
        SlabOperator.__init__(self, rmax= rmax, element_pools=element_pools, NelementLimit = NelementLimit, rng=rng)
        
        self.descriptor = 'RandomElementMutation'
        
    def getName(self):
        return self.__class__.__name__
    
    def get_new_individual(self, inputparents, Ntraits):
        
        Nmutations = Ntraits
        offsprings = []
        for part in inputparents:
            print(f'***Mutation with "{Nmutations}" traits***')
            part = copy.deepcopy(part)
            parentid = part.info['confid']
            # Do the operation
            indi = self.operate(part, Nmutations)
            indi = self.initialize_individual(indi)
            # asssign important info
            indi.info['key_value_pairs']['LastParentID'] = parentid  
            indi.info['key_value_pairs']['LastOperator'] = f'{self.descriptor}_{Nmutations}traits'
            # info about operators
            OperatorGenealogy = indi.info['key_value_pairs']['OperatorGenealogy']
            indi.info['key_value_pairs']['OperatorGenealogy'] = f'{OperatorGenealogy}|{self.descriptor}'
            # info about parents
            Genealogy = indi.info['key_value_pairs']['Genealogy']
            indi.info['key_value_pairs']['Genealogy'] = f'{Genealogy}|{parentid}'
            
            offsprings.append(indi)
        
        return offsprings
    

    def operate(self, a, Nmutations):
        poss_muts = self.get_all_element_mutations(a)
        minlength = min([len(a), len(poss_muts), Nmutations])
        Nmutations = minlength 
        
        random.shuffle(poss_muts)
        MutationsSets  = []
        used_numbers = set()
        # pick the permutation 
        for mut in poss_muts :
            ind = [mut[0]]
            if not set(ind) & used_numbers:  # Check if the combo shares numbers with already used numbers
                MutationsSets.append(mut)
                used_numbers.update(ind)
                
            if len(MutationsSets) == Nmutations:  # Stop once we have 3 unique sets
                break
        
        for mut in MutationsSets:
            
            replace_element(a, mut[0], mut[1], mut[2])
        
        self.dcf(a)
        
        return a


class RandomSurfaceElementMutation(SlabOperator):
    
    '''Mutate elements in one slab
       Accomodate multi slabs in a list 
    '''
    def __init__(self, element_pools = None, NelementLimit = None, rmax = 4.25, rng=np.random):
        
        SlabOperator.__init__(self, rmax= rmax, element_pools=element_pools, NelementLimit = NelementLimit, rng=rng)
        
        self.descriptor = 'RandomElementMutation'
        
    def getName(self):
        return self.__class__.__name__
    
    def get_new_individual(self, inputparents, Ntraits):
        
        Nmutations = Ntraits
        offsprings = []
        for part in inputparents:
            print(f'***SurfaceMutation with "{Nmutations}" traits***')
            part = copy.deepcopy(part)
            parentid = part.info['confid']
            # Do the operation
            indi = self.operate(part, Nmutations)
            indi = self.initialize_individual(indi)
            # asssign important info
            indi.info['key_value_pairs']['LastParentID'] = parentid  
            indi.info['key_value_pairs']['LastOperator'] = f'{self.descriptor}_{Nmutations}traits'
            # info about operators
            OperatorGenealogy = indi.info['key_value_pairs']['OperatorGenealogy']
            indi.info['key_value_pairs']['OperatorGenealogy'] = f'{OperatorGenealogy}|{self.descriptor}'
            # info about parents
            Genealogy = indi.info['key_value_pairs']['Genealogy']
            indi.info['key_value_pairs']['Genealogy'] = f'{Genealogy}|{parentid}'
            
            offsprings.append(indi)
        
        return offsprings
    

    def operate(self, a, Nmutations):
        
        poss_muts = self.get_all_surface_element_mutations(a)
        minlength = min([len(a), len(poss_muts), Nmutations])
        Nmutations = minlength 
        
        random.shuffle(poss_muts)
        MutationsSets  = []
        used_numbers = set()
        
        # pick the permutation 
        for mut in poss_muts :
            ind = [mut[0]]
            if not set(ind) & used_numbers:  # Check if the combo shares numbers with already used numbers
                MutationsSets.append(mut)
                used_numbers.update(ind)
                
            if len(MutationsSets) == Nmutations:  # Stop once we have 3 unique sets
                break
        
        for mut in MutationsSets:
            replace_element(a, mut[0], mut[1], mut[2])
        
        self.dcf(a)
        
        return a
    
    
class RandomElementPermutation(SlabOperator):
    '''Permutate two elemeny in the same slab
       Accomodate multi slabs in a list
    '''
    def __init__(self, element_pools = None, rmax = 4.25, rng=np.random):
        
        SlabOperator.__init__(self, rmax= rmax, element_pools=element_pools,rng=rng)

        self.descriptor = 'RandomElementPermutation'
        
    def getName(self):
        return self.__class__.__name__
    
    def get_new_individual(self, inputparents, Ntraits):
        
        Npermutations = Ntraits
        print(f'***Self-Permutation with "{Npermutations}" traits***')
        # Permutation only makes sense if two different elements are present
        offsprings = []
        for part in inputparents:
            
            part = copy.deepcopy(part)
            parentid = part.info['confid']
            # Do the operation
            indi = self.operate(part, Npermutations)
            indi = self.initialize_individual(indi)
            # asssign important info
            indi.info['key_value_pairs']['LastParentID'] = parentid  
            indi.info['key_value_pairs']['LastOperator'] = f'{self.descriptor}_{Npermutations}traits'
            # info about operators
            OperatorGenealogy = indi.info['key_value_pairs']['OperatorGenealogy']
            indi.info['key_value_pairs']['OperatorGenealogy'] = f'{OperatorGenealogy}|{self.descriptor}'
            # info about parents
            Genealogy = indi.info['key_value_pairs']['Genealogy']
            indi.info['key_value_pairs']['Genealogy'] = f'{Genealogy}|{parentid}'
            
            offsprings.append(indi)
        
        return offsprings
    
    
    def operate(self, atoms, Npermutations):
        # Do the operation
        # Get all possible permutations, ex [1,2], [1,3], [1,4] 
        # the permutations exclude the one with same element between two atoms
        poss_permuts = self.get_all_element_permutations(atoms)
        
        atomslength = len(atoms) 
        if Npermutations >= atomslength:
            Npermutations = atomslength
            
        random.shuffle(poss_permuts)
        picked_sets = []
        used_numbers = set()
        # pick the permutation 
        for ind in poss_permuts:
            if not set(ind) & used_numbers:  # Check if the combo shares numbers with already used numbers
                picked_sets.append(ind)
                used_numbers.update(ind)
            if len(picked_sets) == Npermutations:  # Stop once we have 3 unique sets
                break
            
        Npermutations = picked_sets
        if len(Npermutations) != 0:
            for permuind in Npermutations:
                print(f'atom {permuind[0]} <-> {permuind[1]} | {atoms[permuind[0]].symbol} <-> {atoms[permuind[1]].symbol}')
                permute1(atoms, permuind)
        else:
            print(atoms.get_chemical_formula())
            print(f'{len(Npermutations)} pairs for permutation')
        self.dcf(atoms)
    
        return atoms
    
    
class RandomElementPermutation_2Slabs(SlabOperator):
    '''Permutate two elemeny between 2 slabs
       Accomodate multi slabs in a list
    '''

    def __init__(self, element_pools = None, rmax = 4.25, rng=np.random):
        
        SlabOperator.__init__(self, rmax= rmax, element_pools=element_pools,rng=rng)

        self.descriptor = 'RandomElementPermutation_2Slabs'
        
    def getName(self):
        return self.__class__.__name__
    
    def get_new_individual(self, inputparents, Ntraits):
        Npermutations = Ntraits
        print(f'***2 slabs Permutation with "{Npermutations}" traits***')
        # Permutation only makes sense if two different elements are present
        indiA = copy.deepcopy(inputparents[0])
        indiB = copy.deepcopy(inputparents[1])
        parentidA = indiA.info['confid']
        parentidB = indiB.info['confid']
        
        '''a list of two slabs after permutations'''
        indi = self.operate(indiA, indiB, Npermutations) 
        
        offspring1 = self.initialize_individual(indi[0])
        offspring2 = self.initialize_individual(indi[1])
        
        # asssign important info
        offspring1.info['key_value_pairs']['LastParentID'] = parentidA
        offspring1.info['key_value_pairs']['LastOperator'] = f'{self.descriptor}_{Npermutations}traits'
        # info about operators
        OperatorGenealogy = offspring1.info['key_value_pairs']['OperatorGenealogy']
        offspring1.info['key_value_pairs']['OperatorGenealogy'] = f'{OperatorGenealogy}|{self.descriptor}'
        # info about parents
        Genealogy = offspring1.info['key_value_pairs']['Genealogy']
        offspring1.info['key_value_pairs']['Genealogy'] = f'{Genealogy}|{parentidA}'
        
        # asssign important info
        offspring2.info['key_value_pairs']['LastParentID'] = parentidB
        offspring2.info['key_value_pairs']['LastOperator'] = f'{self.descriptor}_{Npermutations}traits'
        # info about operators
        OperatorGenealogy = offspring2.info['key_value_pairs']['OperatorGenealogy']
        offspring2.info['key_value_pairs']['OperatorGenealogy'] = f'{OperatorGenealogy}|{self.descriptor}'
        # info about parents
        Genealogy = offspring2.info['key_value_pairs']['Genealogy']
        offspring2.info['key_value_pairs']['Genealogy'] = f'{Genealogy}|{parentidB}'

        offsprings = [offspring1, offspring2]
    
        return offsprings
    

    def operate(self, atoms1, atoms2, Npermutations):
        # Do the operation
        # Get all possible permutations, ex [1,2], [1,3], [1,4] 
        # the permutations exclude the one with same element between two atoms
        
        poss_permuts = self.get_all_element_permutations_2slabs(atoms1, atoms2)
        atomslength1 = len(atoms1) 
        atomslength2 = len(atoms2) 
        
        atomslength = min([atomslength1,atomslength2])
        if Npermutations >= atomslength:
            Npermutations = atomslength
            
        random.shuffle(poss_permuts)
        
        picked_sets = []
        used_numbers1 = set()
        used_numbers2 = set()
        
        for ind in poss_permuts:
            num1 = [ind[0]]
            num2 = [ind[1]]
            
            if not set(num1) & used_numbers1 :  # Check if the combo shares numbers with already used numbers
                if not set(num2) & used_numbers2:
                    picked_sets.append(ind)
                    used_numbers1.update(num1)
                    used_numbers2.update(num2)
                
            if len(picked_sets) == Npermutations:  # Stop once we have 3 unique sets
                break
            
        Npermutations = picked_sets
        
        for permuind in Npermutations:
            print(f'atoms1 {permuind[0]} <-> atoms2 {permuind[1]} | {atoms1[permuind[0]].symbol} <-> {atoms2[permuind[1]].symbol}')
            permute2(atoms1, atoms2, permuind)
            
        self.dcf(atoms1)
        self.dcf(atoms2)
        
        return [atoms1,atoms2]
    
class RandommMelting_2Slabs(SlabOperator):
    '''Combine two slabs with new element atom index
    '''
        
    def __init__(self, element_pools = None, rmax = 4.25, rng=np.random):
        
        SlabOperator.__init__(self, rmax= rmax, element_pools=element_pools,rng=rng)

        self.descriptor = 'RandommMelting_2Slabs'

    def getName(self):
        return self.__class__.__name__

    def get_new_individual(self, inputparents, Ntraits):
        print('***2 slabs melting***')
        indiA = copy.deepcopy(inputparents[0])
        indiB = copy.deepcopy(inputparents[1])
        
        parentidA = indiA.info['confid']
        parentidB = indiB.info['confid']
        
        indi, indifacet = self.operate(indiA, indiB) 
        indi = self.initialize_individual(indi)
        
        # asssign important info
        indi.info['key_value_pairs']['facet'] = indifacet
        indi.info['key_value_pairs']['LastParentID'] = f'{parentidA}+{parentidB}'  
        indi.info['key_value_pairs']['LastOperator'] = f'{self.descriptor}'
        # info about operators
        OperatorGenealogy = indi.info['key_value_pairs']['OperatorGenealogy']
        indi.info['key_value_pairs']['OperatorGenealogy'] = f'{OperatorGenealogy}|{self.descriptor}'
        # info about parents
        Genealogy = indi.info['key_value_pairs']['Genealogy']
        indi.info['key_value_pairs']['Genealogy'] = f'{Genealogy}|{parentidA}+{parentidB}'
        
    
        return [indi]
    
    def operate(self, atoms1, atoms2):
        # Do the operation
        poss_syms1 = atoms1.get_chemical_symbols()
        poss_syms2 = atoms2.get_chemical_symbols()
        allsyms = poss_syms1 + poss_syms2
        
        parentfacet1 = atoms1.info['key_value_pairs']['facet']
        parentfacet2 = atoms2.info['key_value_pairs']['facet']
        parentsfacets = [parentfacet1, parentfacet2]
        print(f'Facet 1 : {parentfacet1}')
        print(f'Facet 2 : {parentfacet2}')
        
        '''chance to select slab method
        parent structure should be more favorable
        '''
        Slabs_fcc111 = fcc111('Au', size=(4, 4, 3), a = 3.5, vacuum=10, orthogonal=False, periodic=True)
        Slabs_fcc100 = fcc100('Au', size=(4, 4, 3), a = 3.5, vacuum=10, orthogonal=True, periodic=True)
        Slabs_fcc110 = fcc110('Au', size=(4, 5, 3), a = 2.5, vacuum=10, orthogonal=True, periodic=True)
        Slabs_bcc100 = bcc100('Au', size=(4, 4, 3), a = 2.5, vacuum=10, orthogonal=True, periodic=True)
        Slabs_bcc110 = bcc110('Au', size=(4, 6, 3), a = 3.0, vacuum=10, orthogonal=True, periodic=True)
        Slabs_bcc111 = bcc111('Au', size=(2, 2, 3), a = 4.5, vacuum=10, orthogonal=True, periodic=True)
        Slabs_hcp10m10 = hcp10m10('Au', size=(5, 6, 2), a = 2, vacuum=10, orthogonal=True, periodic=True) 
        Slabs_hcp0001 = hcp0001('Au', size=(4, 4, 3), a = 3, vacuum=10, orthogonal=True, periodic=True)
        
        slabdict = {'fcc111':Slabs_fcc111, 'fcc100':Slabs_fcc100,
                    'fcc110':Slabs_fcc110, 'bcc100':Slabs_bcc100, 
                    'bcc110':Slabs_bcc110, 'bcc111':Slabs_bcc111, 
                    'hcp10m10':Slabs_hcp10m10, 'hcp0001':Slabs_hcp0001} 
        
        slabkeys = list(slabdict.keys())
        weights = [1,0,0,0,0,0,0,0]
        for i in parentsfacets:
            keyind = slabkeys.index(i)
            weights[keyind] += 2
        selectedslab = random.choices(slabkeys, weights=weights, k=1)
        exampleslab = slabdict[str(selectedslab[0])]
        random.shuffle(allsyms)
        newsysms = random.choices(allsyms, k = len(exampleslab))
        exampleslab.set_chemical_symbols(newsysms)
        
        print(f'Slab Likelyhood : fcc111 {weights[0]}, fcc100 {weights[1]}, fcc110 {weights[2]}, bcc100 {weights[3]}, bcc110 {weights[4]}, bcc111 {weights[5]}, hcp10m10 {weights[6]}, hcp0001 {weights[7]}')
        print(f'{parentfacet1} {atoms1.get_chemical_formula()} + {parentfacet2} {atoms2.get_chemical_formula()} -> {selectedslab[0]} {exampleslab.get_chemical_formula()}')
        self.dcf(exampleslab)
        
        return exampleslab, selectedslab[0]
    
    
    
class CutSpliceSlabCrossover(SlabOperator):
    
    def __init__(self, rmax = 4.25, tries=1000, min_ratio=0.25, rng=np.random):
        
        SlabOperator.__init__(self, rmax= rmax,rng=rng)
        self.rng =rng
        self.tries = tries
        self.min_ratio = min_ratio
        self.descriptor = 'CutSpliceSlabCrossover'
        
    def getName(self):
        return self.__class__.__name__
    
    def get_new_individual(self, parents, Ntraits):
        
        print('***CutSpliceSlabCrossover***')

        major = copy.deepcopy(parents[0])
        minor = copy.deepcopy(parents[1])
        parentid_major = major.info['confid']
        parentid_minor = minor.info['confid']
        
        offspring1,offspring2 = self.operate(major, minor)
        indi1 = self.initialize_individual(offspring1)
        indi2 = self.initialize_individual(offspring2)
        
        # asssign important info
        indi1.info['key_value_pairs']['LastParentID'] = f'{parentid_major}+{parentid_minor}'
        indi1.info['key_value_pairs']['LastOperator'] = f'{self.descriptor}'
        # info about operators
        OperatorGenealogy = indi1.info['key_value_pairs']['OperatorGenealogy']
        indi1.info['key_value_pairs']['OperatorGenealogy'] = f'{OperatorGenealogy}|{self.descriptor}'
        # info about parents
        Genealogy = indi1.info['key_value_pairs']['Genealogy']
        indi1.info['key_value_pairs']['Genealogy'] = f'{Genealogy}|{parentid_major}+{parentid_minor}'
        
        # asssign important info
        indi2.info['key_value_pairs']['LastParentID'] = f'{parentid_major}+{parentid_minor}'
        indi2.info['key_value_pairs']['LastOperator'] = f'{self.descriptor}'
        # info about operators
        OperatorGenealogy = indi2.info['key_value_pairs']['OperatorGenealogy']
        indi2.info['key_value_pairs']['OperatorGenealogy'] = f'{OperatorGenealogy}|{self.descriptor}'
        # info about parents
        Genealogy = indi2.info['key_value_pairs']['Genealogy']
        indi2.info['key_value_pairs']['Genealogy'] = f'{Genealogy}|{parentid_major}+{parentid_minor}'
        
        print(f"{major.info['key_value_pairs']['facet']} {major.get_chemical_formula()} + {minor.info['key_value_pairs']['facet']} {minor.get_chemical_formula()} -> {indi1.info['key_value_pairs']['facet']} {indi1.get_chemical_formula()} + {indi2.info['key_value_pairs']['facet']} {indi2.get_chemical_formula()}")
        
        return [indi1, indi2]


    def operate(self, f, m):
        
        parent1syms = copy.deepcopy(f).get_chemical_symbols()
        parent2syms = copy.deepcopy(m).get_chemical_symbols()
        child1 = copy.deepcopy(f)
        child2 = copy.deepcopy(m)
        maxlength = min([len(parent1syms), len(parent2syms)])
        fp = f.positions
        ma = np.max(fp.transpose(), axis=1)
        mi = np.min(fp.transpose(), axis=1)

        for _ in range(self.tries):
            # Find center point of cut
            rv = [self.rng.random() for _ in range(3)]  # random vector
            midpoint = (ma - mi) * rv + mi
            # Determine cut plane
            theta = self.rng.random() * 2 * np.pi  # 0,2pi
            phi = self.rng.random() * np.pi  # 0,pi
            e = np.array((np.sin(phi) * np.cos(theta),
                          np.sin(theta) * np.sin(phi),
                          np.cos(phi)))

            # Cut structures
            d2fp = np.dot(fp - midpoint, e)
            # print(d2fp)
            fpart = d2fp > 0
            # print(fpart)
            ratio = float(np.count_nonzero(fpart)) / len(f)
            
            if ratio < self.min_ratio or ratio > 1 - self.min_ratio:
                continue
            
            for c, i in enumerate(fpart[:maxlength]):
                if i:
                    pass
                    # syms.append(parent1syms[c])
                else:
                    child1[c].symbol = parent2syms[c]
                    # syms.append(parent2syms[c])
      
            
            # print('Symbol existance :')
            # print(fpart)
            # print("Parents'elements :")
            # print(parent1syms)
            # print(parent2syms)

            break 
        
        child1syms = child1.get_chemical_symbols()
        # identify the difference between child1 and parent1 + parent2
        ind_diff_elements_p1 = [i for i in range(min(len(parent1syms), len(child1syms))) if parent1syms[i] != child1syms[i]]
        ind_diff_elements_p2 = [i for i in range(min(len(parent2syms), len(child1syms))) if parent2syms[i] != child1syms[i]]
        # print(ind_diff_elements_p2)
        # assign the element in to child2 
        for indp1 in ind_diff_elements_p1:
            child2[indp1].symbol = parent1syms[indp1]
        for indp2 in ind_diff_elements_p2:
            child2[indp2].symbol = parent2syms[indp2]
            
        self.dcf(child1)
        self.dcf(child2)
        
        return [child1, child2]


class OperationSelector:
    """Class used to randomly select a procreation operation
    from a list of operations.

    Parameters:

    probabilities: A list of probabilities with which the different
        mutations should be selected. The norm of this list
        does not need to be 1.

    oplist: The list of operations to select from.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, probabilities, oplist, rng=np.random):
        
        assert len(probabilities) == len(oplist)
        
        self.oplist = oplist
        self.rho = np.cumsum(probabilities)
        self.rng = rng
        self.oplistname = [i.getName()for i in self.oplist]
        for c, op in enumerate(self.oplistname):
            print(f'Operator {c} : {op} | {probabilities[c]}')


    def __get_index__(self):
        v = self.rng.random() * self.rho[-1]
        for i in range(len(self.rho)):
            if self.rho[i] > v:
                return i

    def get_new_individual(self, candidate_list):
        """Choose operator and use it on the candidate. """
        to_use = self.__get_index__()
        return self.oplist[to_use].get_new_individual(candidate_list)

    def get_operator(self):
        """Choose operator and return it."""
        to_use = self.__get_index__()
        # print(f'-----{self.oplistname[to_use]} is selected')
        return self.oplist[to_use], self.oplistname[to_use]
    
    
    
    
'''Operator Debug'''
# element_pools = ['Pd', 'Ni']
# metals = ['Au', 'Fe']
# MetalSlabs_fcc111 = [fcc111(m, size=(4,4,3), a = 3.5, vacuum=10, orthogonal=False, periodic=True) for m in metals]
# MetalSlabs_bcc100 = [bcc100(m, size=(4,4,3), a = 2.5, vacuum=10, orthogonal=True, periodic=True) for m in metals]

# A = SurfaceAtomFinder(MetalSlabs_fcc111[0])
# c1, c2 = MetalSlabs_bcc100[0], MetalSlabs_fcc111[1]
# c1.info['confid'] = '12345'
# c2.info['confid'] = '45678'
# Operator = SlabOperator()
# c1 = Operator.initialize_individual(c1, facet = 'fcc111')
# c2 = Operator.initialize_individual(c2, facet = 'bcc100')

# c1.info['key_value_pairs']['facet'] = 'fcc111'
# c2.info['key_value_pairs']['facet'] = 'bcc100'

# parents = [c1, c2]
# '''After the operator, needs to update the id for new structure'''
# c1info = c1.info
# c2info = c2.info

# Duplicate identification



#RandommMelting_2Slabs
# op = RandommMelting_2Slabs()
# output = op.get_new_individual(parents)
# view(output)

#RandomElementPermutation_2Slabs
# op = RandomElementPermutation_2Slabs(num_muts=2)
# output = op.get_new_individual(parents)

#RandomElementMutation
# op = RandomElementMutation(element_pools = element_pools, num_muts=10)
# output = op.get_new_individual(parents)
# view(output)

#RandomSlabPermutation
# op = RandomElementPermutation(num_muts=3)
# output2 = op.get_new_individual(output)

#CutSpliceSlabCrossover
# op = CutSpliceSlabCrossover()
# output1 = op.get_new_individual(output)















