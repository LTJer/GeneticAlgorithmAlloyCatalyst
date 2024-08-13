import numpy as np
from catkit.gen.adsorption import AdsorptionSites, _get_adsorption_sites
from catkit import Gratoms
from statistics import mean
from ase import Atom, Atoms
import warnings
import pandas as pd
from pymatgen.core import Lattice
from pymatgen.analysis.adsorption import put_coord_inside
import scipy

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def expand_cell(atoms, cutoff=None, padding=None):
    """Return Cartesian coordinates atoms within a supercell
    which contains repetitions of the unit cell which contains
    at least one neighboring atom.

    Parameters
    ----------
    atoms : Atoms object
        Atoms with the periodic boundary conditions and unit cell
        information to use.
    cutoff : float
        Radius of maximum atomic bond distance to consider.
    padding : ndarray (3,)
        Padding of repetition of the unit cell in the x, y, z
        directions. e.g. [1, 0, 1].

    Returns
    -------
    index : ndarray (N,)
        Indices associated with the original unit cell positions.
    coords : ndarray (N, 3)
        Cartesian coordinates associated with positions in the
        supercell.
    offsets : ndarray (M, 3)
        Integer offsets of each unit cell.
    """
    cell = atoms.cell
    pbc = atoms.pbc
    pos = atoms.positions

    if padding is None and cutoff is None:
        diags = np.sqrt((
            np.dot([[1, 1, 1],
                    [-1, 1, 1],
                    [1, -1, 1],
                    [-1, -1, 1]],
                   cell)**2).sum(1))

        if pos.shape[0] == 1:
            cutoff = max(diags) / 2.
        else:
            dpos = (pos - pos[:, None]).reshape(-1, 3)
            Dr = np.dot(dpos, np.linalg.inv(cell))
            D = np.dot(Dr - np.round(Dr) * pbc, cell)
            D_len = np.sqrt((D**2).sum(1))

            cutoff = min(max(D_len), max(diags) / 2.)

    latt_len = np.sqrt((cell**2).sum(1))
    V = abs(np.linalg.det(cell))
    padding = pbc * np.array(np.ceil(cutoff * np.prod(latt_len) /
                                     (V * latt_len)), dtype=int)

    offsets = np.mgrid[
        -padding[0]:padding[0] + 1,
        -padding[1]:padding[1] + 1,
        -padding[2]:padding[2] + 1].T
    tvecs = np.dot(offsets, cell)
    coords = pos[None, None, None, :, :] + tvecs[:, :, :, None, :]

    ncell = np.prod(offsets.shape[:-1])
    index = np.arange(len(atoms))[None, :].repeat(ncell, axis=0).flatten()
    coords = coords.reshape(np.prod(coords.shape[:-1]), 3)
    offsets = offsets.reshape(ncell, 3)

    return index, coords, offsets




def get_voronoi_neighbors(atoms, cutoff=3.0, return_distances=False):
    """Return the connectivity matrix from the Voronoi
    method. Multi-bonding occurs through periodic boundary conditions.

    Parameters
    ----------
    atoms : atoms object
        Atoms object with the periodic boundary conditions and
        unit cell information to use.
    cutoff : float
        Radius of maximum atomic bond distance to consider.

    Returns
    -------
    connectivity : ndarray (n, n)
        Number of edges formed between atoms in a system.
    """
    index, coords, offsets = expand_cell(atoms, cutoff=cutoff)

    xm, ym, zm = np.max(coords, axis=0) - np.min(coords, axis=0)

    L = int(len(offsets) / 2)
    origional_indices = np.arange(L * len(atoms), (L + 1) * len(atoms))

    voronoi = scipy.spatial.Voronoi(coords, qhull_options='QbB Qc Qs')
    points = voronoi.ridge_points

    connectivity = np.zeros((len(atoms), len(atoms)))
    distances = []
    distance_indices = []
    for i, n in enumerate(origional_indices):
        ridge_indices = np.where(points == n)[0]
        p = points[ridge_indices]
        dist = np.linalg.norm(np.diff(coords[p], axis=1), axis=-1)[:, 0]
        edges = np.sort(index[p])

        if not edges.size:
            warnings.warn(
                ("scipy.spatial.Voronoi returned an atom which has "
                 "no neighbors. This may result in incorrect connectivity."))
            continue

        unique_edge = np.unique(edges, axis=0)

        for j, edge in enumerate(unique_edge):
            indices = np.where(np.all(edge == edges, axis=1))[0]
            d = dist[indices][np.where(dist[indices] < cutoff)[0]]
            count = len(d)
            if count == 0:
                continue

            u, v = edge

            distance_indices += [sorted([u, v])]
            distances += [sorted(d)]

            connectivity[u][v] += count
            connectivity[v][u] += count

    connectivity /= 2
    if not return_distances:
        return connectivity.astype(int)

    if len(distances) > 0:
        distance_indices, unique_idx_idx = \
            np.unique(distance_indices, axis=0, return_index=True)
        distance_indices = distance_indices.tolist()

        distances = [distances[i] for i in unique_idx_idx]

    pair_distances = {'indices': distance_indices,
                      'distances': distances}

    return connectivity.astype(int), pair_distances


def id_surface_atoms(atoms, classifier='voronoi_sweep', VoronoiCutoff=4.0):
        """Identify surface atoms of an atoms object. This will
        require that adsorbate atoms have already been identified.

        Parameters
        ----------
        classifier : str
            Classification technique to identify surface atoms.

            'voronoi_sweep':
            Create a sweep of proxy atoms above surface. Surface atoms
            are those which are most frequent neighbors of the sweep.

        Returns
        -------
        surface_atoms : ndarray (n,)
            Index of the surface atoms in the object.
        """
        atoms = atoms.copy()
        
        if classifier == 'voronoi_sweep':
            spos = atoms.get_scaled_positions()
            zmax = np.max(spos[:, -1])

            # Create a distribution of points to screen with
            # 2.5 angstrom defines the absolute separation
            absseparation = 2.5
            dvec = (np.linalg.norm(atoms.cell[:-1], axis=1) / absseparation) ** -1
            xy = np.mgrid[0:1:dvec[0], 0:1:dvec[1]].reshape(2, -1)
            z = np.ones_like(xy[0]) * zmax
            xyz = np.vstack((xy, z)).T

            screen = np.dot(xyz, atoms.cell)

            n = len(atoms)
            m = len(screen)
            ind = np.arange(n, n + m)

            slab_atoms = np.arange(n)

            satoms = []
            # 2 - 3 Angstroms seems to work for a large range of indices.
            for k in np.linspace(2, 3, 10):
                wall = screen.copy() + [0, 0, k]

                atm = Atoms(['X'] * m, positions=wall)
                test_atoms = atoms + atm

                con = get_voronoi_neighbors(atoms = test_atoms, cutoff=VoronoiCutoff)
                surf_atoms = np.where(con[ind].sum(axis=0)[slab_atoms])[0]
                satoms += [surf_atoms]
            len_surf_atoms = [len(_) for _ in satoms]
            uni, ind, cnt = np.unique(
                len_surf_atoms, return_counts=True, return_index=True)

            max_cnt = np.argmax(cnt)
            surf_atoms = satoms[ind[max_cnt]]

        return surf_atoms
    
    

def norm(a1,a2,a3):
    return (a1**2 + a2**2 + a3**2)**0.5


def get_distance_perodic_cartesian(vec1, vec2, unit_cell):
    # calculating distances of input fractional coordinate
    delta_a = vec1[0] - vec2[0]
    delta_b = vec1[1] - vec2[1]
    delta_c = vec1[2] - vec2[2]

    if delta_a > 0.5:
       vec2[0] = vec2[0] + 1.0
    elif delta_a < -0.5:
       vec2[0] = vec2[0] - 1.0

    if delta_b > 0.5:
       vec2[1] = vec2[1] + 1.0
    elif delta_b < -0.5:
       vec2[1] = vec2[1] - 1.0

    if delta_c > 0.5:
       vec2[2] = vec2[2] + 1.0
    elif delta_c < -0.5:
       vec2[2] = vec2[2] - 1.0
       
    FractionalDistVector = vec1 - vec2
    CartesianDistVector =  np.dot(np.transpose(unit_cell), FractionalDistVector)
    CaresianDistance = norm(CartesianDistVector[0],CartesianDistVector[1],CartesianDistVector[2])

    return  CaresianDistance

def get_distance_perodic_cartesianS(cartesianP1, cartesianPs, pymlattice):
    fractioncenter = pymlattice.get_fractional_coords(cartesianP1)
    fractionPs = pymlattice.get_fractional_coords(cartesianPs)
    cartesian_distances = []
    for p in fractionPs:
        cartesianDist = get_distance_perodic_cartesian(fractioncenter, p, pymlattice.matrix)
        cartesian_distances.append(cartesianDist)
    cartesian_distances = np.array(cartesian_distances)
    return cartesian_distances



def DelaunayTopologytransfer(surfatoms_list, topology_Delaunay):
    
    surfindex = {}
    for c, k in enumerate(surfatoms_list):
        surfindex[c] = k
        
    topology_Delaunay = np.reshape(topology_Delaunay, (len(topology_Delaunay),1))
    topology_temp = []
    connectivity_Delaunay = []
    for i in topology_Delaunay:
        for j in i:
            list1 = [surfindex[k] for k in j]
        topology_temp.append(list1)
        connectivity_Delaunay.append(len(list1))
    Corrected_topology_Delaunay = np.array(topology_temp,dtype=object).reshape((len(topology_temp),1))
    Corrected_connectivity_Delaunay = np.reshape(connectivity_Delaunay, (len(connectivity_Delaunay),1))
    
    return Corrected_topology_Delaunay, Corrected_connectivity_Delaunay


def FetchBottomAtom_ASE(SiteCoord, slab_ase, xytolerance = 0.3, ztolerance = 4):
    
    neighbors = slab_ase.arrays['positions']
    dx = (neighbors[:,0:2] - SiteCoord[0:2])[:,0]
    dy = (neighbors[:,0:2] - SiteCoord[0:2])[:,1]
    dz = neighbors[:,2] - SiteCoord[2]
    dr_xy = ((dx**2)+(dy**2))**0.5 # the radius on xy surface
    
    #check the atoms with distances shorter than xytolerance 
    XYindex = np.where(dr_xy <= xytolerance)[0]
    #check the atoms with z distances shorter than ztolerance 
    Ztoler = np.where(abs(dz) <= ztolerance)[0]
    #check the atoms is below the site
    Z_negative = np.where(dz <= 0)[0]
    # find the common indices from previous three checker
    XYZindex = list(set(XYindex).intersection(Z_negative,Ztoler))#[i for i, j in zip(Z_negative, Z_toler) if i == j]
    # the atoms belong to adsorbate
    adsindex = np.where((slab_ase.numbers != 13)&(slab_ase.numbers < 21))[0].tolist()
    # find the indices are included in XYZindex but not in adsindex
    final_index = np.setdiff1d(XYZindex, adsindex) # needs to be only one value
    lengh = len(final_index)
    
    if lengh == 1:
        bottomelement = slab_ase.symbols[int(final_index)]
        return bottomelement
    
    elif lengh == 0:
        bottomelement = np.nan
        
    else :
        bottomelement = np.nan
        print('detected {lengh} atoms under the hollow site')
        
    return bottomelement

def DistinctHollowSites_ase(SiteCoord, slab_ase, xytolerance = 0.3, ztolerance = 4):
    
    neighbors = slab_ase.arrays['positions']
    dx = (neighbors[:,0:2] - SiteCoord[0:2])[:,0]
    dy = (neighbors[:,0:2] - SiteCoord[0:2])[:,1]
    dz = neighbors[:,2] - SiteCoord[2]
    dr_xy = ((dx**2)+(dy**2))**0.5 # the radius on xy surface
    
    #check the atoms with distances shorter than xytolerance 
    XYindex = np.where(dr_xy <= xytolerance)[0]
    #check the atoms with z distances shorter than ztolerance 
    Ztoler = np.where(abs(dz) <= ztolerance)[0]
    #check the atoms is below the site
    Z_negative = np.where(dz <= 0)[0]
    # find the common indices from previous three checker
    XYZindex = list(set(XYindex).intersection(Z_negative,Ztoler))#[i for i, j in zip(Z_negative, Z_toler) if i == j]
    # the atoms belong to adsorbate
    adsindex = np.where((slab_ase.numbers != 13)&(slab_ase.numbers < 21))[0].tolist()
    # find the indices are included in XYZindex but not in adsindex
    final_list = np.setdiff1d(XYZindex, adsindex)
    
    if len(final_list) >= 1:
        return 'hcpHollow'
    else:
        return 'fccHollow'


def SiteFinder_Hybride(AseAtoms = 'MetalSlabs_fcc111',
                        adsorbates = ['N'],
                        VoronoiCutoff = 4,
                        Zcoord_add = 1.5,
                        RoundDicimal=1,
                        xytolerance=0.3,
                        ztolerance=4.5,
                        overlaping = 0.7,
                        adsdeviation = 0.8,
                        countvalues = 0,
                        NmaxNeighbor = 5,
                        ClosestSiteFinder = False,
                        edges=None):

    
    '''This function is the latest version for catching active sites on the catalyst surface 
    for both FCC, BCC, HCP (top, bridge, fcc hollow, hcphollow, 4-fold hollow)
    same function as the in DOSnet/BuildMLData/sitefinder_Debug.py "SiteFinder_Hybride_new"
    '''
    
    """
    The function can search the adsorption sites on the given catalyst surface (support both pure metal and alloy) via
    both graphic and delaunay method, https://doi.org/10.1021/acs.jpca.9b00311.
    
    AseAtoms                : slab structure in the type of ase.atoms
    adsorbates              : the adsorbate in on the given slab in a list which will be removed during the site identification
    VoronoiCutoff           : The distance between slab surface and the pseudo atom above the surface
                              used to identify surface atom through voronoi_sweep in function id_surface_atoms()
    Zcoord_add              : the z direction increament for puting the pseudo atoms for visualization, mean_z + z_increament
                              mean_z is the average value of z coordinate of all surface atoms
    RoundDicimal            : the dicimal for rounded the site coordinates
    xytolerance, ztolerance : the tolerance values to identify the bottom atoms of active site, only work for hcp,fcc, and 4-fold hollows
    ClosestSiteFinder       : whether to fine the closest site of the given surface only adsorbate
    """ 
    
    '''Create copy of surface and extended surface for adsorption site identification'''
    origsurface = AseAtoms.copy()
    lattice_pym =  Lattice(origsurface.cell.array, pbc= (True, True, True))
    # remove ads on the surface based on the input paramater "adsorbates"
    cleansurface = origsurface.copy()
    del cleansurface[[atom.index for atom in cleansurface if atom.symbol in adsorbates]]
    # create an extended surface for site searching; extended surface can help to identify the site close to the uncell's boundary
    # where the sites out side of original unitcell will be folding back in the later section 
    Excleansurface = cleansurface.repeat((2, 2, 1))
    # setup graphic ase object based on the extended surface 
    GAtomic_Ex = Gratoms(numbers=Excleansurface.numbers,
                         positions=Excleansurface.positions,
                         pbc=Excleansurface.pbc,
                         cell= Excleansurface.cell,
                         edges=edges)
    # catch surface atoms via Voronoi sweep; the slowest funciton; the surface atom determine the accuracy of site identification
    # the VoronoiCutoff is the key paramater to identify the surface atom; if the surface atom cannot be determined, try reduce the curoff 
    surfatomslist = id_surface_atoms(atoms = Excleansurface, VoronoiCutoff = VoronoiCutoff) 
    surfatomsposs = GAtomic_Ex.positions[surfatomslist]
    # setup Z coordinate for putting adsorbate or visualized pseudo atom  
    site_newZ = mean(surfatomsposs[:,2]) + Zcoord_add
    
    '''Surface adsorption sites from two different methods, Delaunay and Graph theory:
        three important outputs, coordinates, topology(surrounding atoms), and connectivity for each method
    ''' 
    '''Delaunay method'''
        #topology2_Delaunay is the second neighbor, less likely to be used
    coordinates_Delaunay, topology_Delaunay, topology2_Delaunay = _get_adsorption_sites(surfatomsposs, tol=1e-7)
    coordinates_Delaunay[:,2] = site_newZ
    coordinates_Delaunay = put_coord_inside(lattice = lattice_pym, cart_coordinate = coordinates_Delaunay)
        # build the connectivity and check if the length of topology matches surfatomslist
    topology_Delaunay, connectivity_Delaunay = DelaunayTopologytransfer(surfatomslist, topology_Delaunay) 
    
    
    '''Graph theory'''
        # setup surface atoms
    GAtomic_Ex.set_surface_atoms(top=surfatomslist)
        # setup catkit graphic surface site finder
    SiteFinder_Catkit = AdsorptionSites(GAtomic_Ex)
        # call all the site coordinate
    coordinates_Catkit = SiteFinder_Catkit.coordinates
    coordinates_Catkit[:,2] = site_newZ
    coordinates_Catkit = put_coord_inside(lattice = lattice_pym, cart_coordinate = coordinates_Catkit)
        # call all the connectivity if site
    connectivity_Catkit = SiteFinder_Catkit.connectivity.reshape((-1, 1))
        # call all the topology of site; r1 meand the first neighbors
    topology_Catkit = np.array([SiteFinder_Catkit.index[top] for top in SiteFinder_Catkit.r1_topology]).reshape((-1, 1))
    
    '''Combine topoloies, connectivities, and coordinates from both methods, separately
       Find the indices of unique coordinates; then use the indices to exclude the replicated sites in the three objects
       Notice that: put the Delaunay before the Catkit in "np.vastack" can increas the importance and help identify the 4-fold hollow.
    ''' 
    coordinates_sum = np.vstack((coordinates_Delaunay,coordinates_Catkit))
    # identify and exclude the sites with exactly the same coordinates 
    uniindex = np.unique(coordinates_sum.round(decimals=RoundDicimal), axis = 0, return_index=True)[1]
    coordinates_sum = coordinates_sum[np.sort(uniindex)]
    # combine topologies
    topology_sum = np.vstack((topology_Delaunay,topology_Catkit))[np.sort(uniindex)]
    # combine connectivities
    connectivity_sum = np.vstack((connectivity_Delaunay,connectivity_Catkit,))[np.sort(uniindex)]
    
    '''Find and exclude the overlaping sites from the slab out of initial unitcell or the site is too close to each by "overlaping"
       the order of coordinates_sum represents the priority of the sites, 1 > 2 > 4~3...etc
        export important arrays: sort_unicoord, sort_uniconnectivity, and sort_unitopology for further analysis
        
    '''
    # create adjacency matrix of sites
    adjsize = len(coordinates_sum)
    adjmatrix = np.zeros((adjsize,adjsize))
    for row, coord in enumerate(coordinates_sum):
        D_len = get_distance_perodic_cartesianS(cartesianP1 = coord, cartesianPs=coordinates_sum, pymlattice = lattice_pym)
        for col, length in enumerate(D_len.reshape(adjsize,1)):
            adjmatrix[row][col] = length
    # make the upper triangular part as 100 to escape doulbe counting minimum value 
    adjmatrix[np.triu_indices(adjmatrix.shape[0], 0)] = 100
    # minimum value for each row 
    adjmatrix_rowmin = adjmatrix.min(axis=1)
    # filter out the indices with distance between sites less than the "overlaping" criteria in angstrom
    excludelist = np.where(adjmatrix_rowmin <= overlaping)[0].tolist()
    # build the index list for preserving sites and exclude the overlaping sites
    RetainIndex= np.sort(list(set([i for i, _ in enumerate(coordinates_sum)]) - set(excludelist)))
    # use the index to select valid sites
    sort_unicoord = coordinates_sum[RetainIndex]
    sort_uniconnectivity = connectivity_sum[RetainIndex]
    sort_unitopology = topology_sum[RetainIndex]
    
    '''assign neighbor atoms type base on the "sort_unitopology"''' 
    ExtandNeighbor = [] # the neighbor list can be extended from 3 to 5 neighbors
    sitetype = []
    # max number of neighbor should be considered, the empty neighbor will be nan
    NmaxNeighbor = NmaxNeighbor 
    for co, list0 in enumerate(sort_unitopology):
        for num in list0:
            typ = len(num)
            if typ == 4:
                bottomsym = FetchBottomAtom_ASE(sort_unicoord[co], GAtomic_Ex, xytolerance, ztolerance)
                symbol = list(GAtomic_Ex.symbols[num])
                if NmaxNeighbor == 5:
                    symbol.append(bottomsym)
                elif NmaxNeighbor == 4:
                    pass
                sitetype.append('4-fold')
            elif typ == 3:
                bottomsym = FetchBottomAtom_ASE(sort_unicoord[co], GAtomic_Ex, xytolerance, ztolerance)
                symbol = list(GAtomic_Ex.symbols[num])
                symbol += [np.nan]*(NmaxNeighbor-typ-1) # fill empty neighbor with nan
                symbol.append(bottomsym)
                if pd.isna(bottomsym):
                    sitetype.append('fccHollow')
                else:
                    sitetype.append('hcpHollow')
            elif typ == 2 :
                symbol = list(GAtomic_Ex.symbols[num])
                symbol += [np.nan]*(NmaxNeighbor-typ) # fill empty neighbor with nan
                sitetype.append('bridge')
                
            elif typ == 1 :
                symbol = list(GAtomic_Ex.symbols[num])
                symbol += [np.nan]*(NmaxNeighbor-typ)
                sitetype.append('ontop')
            else:
                print('Detected weird topology')
                
            ExtandNeighbor.append(symbol)
            
    ExtandNeighbor = np.array(ExtandNeighbor)
    sitetype = np.array(sitetype).reshape((-1, 1))
    
    '''combine the unique info together for exporting'''
    if NmaxNeighbor == 4:
        columns=['x','y','z', 'connectivity','topology','sitetype','neighbor1','neighbor2','neighbor3','neighbor_bottom']
        
    elif NmaxNeighbor == 5: 
        columns=['x','y','z', 'connectivity','topology','sitetype','neighbor1','neighbor2','neighbor3','neighbor4','neighbor_bottom']
    # export check point for validation
    outputarray = np.hstack((sort_unicoord,sort_uniconnectivity,sort_unitopology,sitetype,ExtandNeighbor))
    outputarray = pd.DataFrame(outputarray, columns = columns)
    
    '''Put pseudo atoms for visulaize sites on the original surface'''
    atm = {'ontop':'O','bridge':'H','hcpHollow': 'F', 'fccHollow': 'C','4-fold':'Ne' }
    adjustment = [0, 0, 0]
    Visualatoms = origsurface.copy()
    for c, pos in enumerate(sort_unicoord):
        sitetyp = atm[sitetype[c][0]]
        Visualatoms += Atom(sitetyp, pos + adjustment)
        
    '''find the closet site and check if the distance is less than adsdeviation criteria
    ## if the adsorbate is too far from the any site identified from the previous part 
        and above the criteria "adsdeviation" then the geometry will be rejected 
    '''
    ClosestAdsDict = {}
    if ClosestSiteFinder:
        
        # find the ads positions, accomodate only one ads
        for atom in origsurface :
            if atom.symbol in adsorbates:
                ads_position = atom.position
                ads_position[2] = site_newZ
        # calculate the distance between existing sites with ads
        ads_distlen = get_distance_perodic_cartesianS(cartesianP1 = ads_position, cartesianPs=sort_unicoord, pymlattice = lattice_pym).reshape(-1,1)
        # find the index of the minimum distance
        minindex = int(np.argmin(ads_distlen, axis=0))
        minvalue = ads_distlen[minindex]
        # clean slab formula
        formula = cleansurface.get_chemical_formula(empirical=False)
        
        if minvalue > adsdeviation:
            print(f'Frame {countvalues} "{formula}" cannot find any matching site')
            print(f'Distance between target site and ads : {minvalue}')
            print(f'export {formula} as empty dictionary')
            
        else:
            # neighbor list of the closest site of the ads
            env = ExtandNeighbor[minindex].tolist()
            # site type of the closest site of the ads
            sitetyp = str(sitetype[minindex][0])
            # correct position of the closest site of the ads
            sitepos = sort_unicoord[minindex]
            # connectivity of the closest site of the ads
            conn = sort_uniconnectivity[minindex].tolist()[0]
            # build a dict wiht all info 
            ClosestAdsDict = {'formula':formula, 'site':sitetyp, 'topology':env, 'sitepos':sitepos, 'connectivity' : conn}
            # Assign a red point to the closest site
            Visualatoms.symbols[len(origsurface.numbers) + minindex] = 'X'
            
    return outputarray, Excleansurface, Visualatoms, ClosestAdsDict



def SurfaceAtomFinder(AseAtoms = 'MetalSlabs_fcc111',                 
                        VoronoiCutoff = 4):

    
    '''Create copy of surface and extended surface for adsorption site identification'''
    origsurface = AseAtoms.copy()
    cleansurface = origsurface.copy()
    surfatomslist = id_surface_atoms(atoms = cleansurface, VoronoiCutoff = VoronoiCutoff) 
            
    return surfatomslist


def ML_Pred_Avg_BindingE(model, slab, DataRescaler, MetalDOS_Dict, verbose = False):
    
    '''Predict the binding energy of both fcc and hcp hollow; than calculate the average:
        model         : pre-train ML model for Binding Energy prediction 
        slab          : sleb to predict the average binding energy
        MetalDOS_Dict : dictionary to provide the bare surface DOS
    '''
    # use universal site finder to find all hollow sites on the catalyst surface 
    df_SiteFinder, Excleansurface, Visualatoms, ClosestAdsDict = \
    SiteFinder_Hybride(AseAtoms = slab, 
                        adsorbates = ['N'],
                        VoronoiCutoff = 4,
                        Zcoord_add = 1.5,
                        RoundDicimal=1,
                        xytolerance=0.3,
                        ztolerance=4.5,
                        overlaping = 0.7,
                        adsdeviation = 0.8,
                        countvalues = 0,
                        NmaxNeighbor = 5,
                        ClosestSiteFinder = False,
                        edges=False)
    
    
    # extract all hollow sites and corresponding neighbors 
    '''Need to include 5 neighbor including bottom atom'''
    fccHollow = df_SiteFinder.loc[df_SiteFinder['sitetype'] == 'fccHollow'][['neighbor1','neighbor2','neighbor3','neighbor4','neighbor_bottom']]
    hcpHollow = df_SiteFinder.loc[df_SiteFinder['sitetype'] == 'hcpHollow'][['neighbor1','neighbor2','neighbor3','neighbor4','neighbor_bottom']]
    fourfold = df_SiteFinder.loc[df_SiteFinder['sitetype'] == '4-fold'][['neighbor1','neighbor2','neighbor3','neighbor4','neighbor_bottom']]
    if verbose:
        print(f'identify {len(fourfold)} 4-fold-hollow sites')
        print(f'identify {len(hcpHollow)} hcp-hollow sites')
        print(f'identify {len(fccHollow)} fcc-hollow sites')
    # print('------------------------------------------')
    # the zero DOS for empty atom
    emptyatom = np.zeros((len(MetalDOS_Dict['Au']),9))
    # collect all DOS for each site together
    SiteDOSs = []
    # loop through all sites and assign clean surface DOSs
    SiteNeighbors = pd.concat([fccHollow, hcpHollow, fourfold],axis=0).to_numpy()
    for topology in SiteNeighbors :
        DOSs = MetalDOS_Dict[topology[0]][:,0:9]
        for i in range(1,len(SiteNeighbors[0])):
            ele = topology[i]
            if ele != 'nan':
                # append bottom atom's DOS
                #s, p_y, p_z, p_x, d_xy, d_yz, d_z2-r2, d_xz, d_x2-y2 ## no f orbital
                DOS = MetalDOS_Dict[ele][:,0:9]
                DOSs = np.hstack((DOSs,DOS))
            else:
                DOSs = np.hstack((DOSs,emptyatom))
        SiteDOSs.append(DOSs)    
        
    SiteDOSs = np.array(SiteDOSs)
    # print(np.shape(SiteDOSs))
    # rescale the DOS values for machine learning model prediction and discard the dos after 2000
    MLinput = SiteDOSs[:, 0:2000, 0:45]
    MLinput = DataRescaler.transform(MLinput.reshape(-1, MLinput.shape[2])).reshape(MLinput.shape)
    # predict1 the binding energy for each site
    BindingE = model.predict([MLinput[:, :, 0:9], MLinput[:, :, 9:18], MLinput[:, :, 18:27], MLinput[:, :, 27:36], MLinput[:, :, 36:45]], 
                             verbose = 0)
    # mean of binding energy of all sites 
    ExportBindingE = np.mean(BindingE)
    print(f'Avg binding energy {round(ExportBindingE,4)} from {len(BindingE)} sites') 
    return ExportBindingE, Visualatoms

def ML_Pred_Highest_BindingE(model, slab, DataRescaler, MetalDOS_Dict,verbose = False):
    
    '''Predict the binding energy of both fcc and hcp hollow; than calculate the average:
        model         : pre-train ML model for Binding Energy prediction 
        slab          : sleb to predict the average binding energy
        MetalDOS_Dict : dictionary to provide the bare surface DOS
    '''
    # use universal site finder to find all hollow sites on the catalyst surface 
    df_SiteFinder, Excleansurface, Visualatoms, ClosestAdsDict = \
    SiteFinder_Hybride(AseAtoms = slab, 
                        adsorbates = ['N'],
                        VoronoiCutoff = 4,
                        Zcoord_add = 1.5,
                        RoundDicimal=1,
                        xytolerance=0.3,
                        ztolerance=4.5,
                        overlaping = 0.7,
                        adsdeviation = 0.8,
                        countvalues = 0,
                        NmaxNeighbor = 5,
                        ClosestSiteFinder = False,
                        edges=False)
    
    
    # extract all hollow sites and corresponding neighbors 
    '''Need to include 5 neighbor including bottom atom'''
    fccHollow = df_SiteFinder.loc[df_SiteFinder['sitetype'] == 'fccHollow'][['neighbor1','neighbor2','neighbor3','neighbor4','neighbor_bottom']]
    hcpHollow = df_SiteFinder.loc[df_SiteFinder['sitetype'] == 'hcpHollow'][['neighbor1','neighbor2','neighbor3','neighbor4','neighbor_bottom']]
    fourfold = df_SiteFinder.loc[df_SiteFinder['sitetype'] == '4-fold'][['neighbor1','neighbor2','neighbor3','neighbor4','neighbor_bottom']]
    if verbose:
        print(f'identify {len(fourfold)} 4-fold-hollow sites')
        print(f'identify {len(hcpHollow)} hcp-hollow sites')
        print(f'identify {len(fccHollow)} fcc-hollow sites')
    # print('------------------------------------------')
    # the zero DOS for empty atom
    emptyatom = np.zeros((len(MetalDOS_Dict['Au']),9))
    # collect all DOS for each site together
    SiteDOSs = []
    # loop through all sites and assign clean surface DOSs
    SiteNeighbors = pd.concat([fccHollow, hcpHollow, fourfold],axis=0).to_numpy()
    for topology in SiteNeighbors :
        DOSs = MetalDOS_Dict[topology[0]][:,0:9]
        for i in range(1,len(SiteNeighbors[0])):
            ele = topology[i]
            if ele != 'nan':
                # append bottom atom's DOS
                #s, p_y, p_z, p_x, d_xy, d_yz, d_z2-r2, d_xz, d_x2-y2 ## no f orbital
                DOS = MetalDOS_Dict[ele][:,0:9]
                DOSs = np.hstack((DOSs,DOS))
            else:
                DOSs = np.hstack((DOSs,emptyatom))
        SiteDOSs.append(DOSs)    
        
    SiteDOSs = np.array(SiteDOSs)
    # print(np.shape(SiteDOSs))
    # rescale the DOS values for machine learning model prediction and discard the dos after 2000
    MLinput = SiteDOSs[:, 0:2000, 0:45]
    MLinput = DataRescaler.transform(MLinput.reshape(-1, MLinput.shape[2])).reshape(MLinput.shape)
    # predict1 the binding energy for each site
    BindingE = model.predict([MLinput[:, :, 0:9], MLinput[:, :, 9:18], MLinput[:, :, 18:27], MLinput[:, :, 27:36], MLinput[:, :, 36:45]], 
                             verbose = 0)
    # mean of binding energy of all sites
    ExportBindingE = np.max(BindingE)
    print(f'Highest binding energy {round(ExportBindingE,4)} from {len(BindingE)} sites')
    return ExportBindingE, Visualatoms



def get_avg_lattice_constant(syms, lattice_constants):
    # predict the lattice constant via via Vegard's law
    a = 0.
    for m in set(syms):
        a += syms.count(m) * lattice_constants[m]
    return a / len(syms)
