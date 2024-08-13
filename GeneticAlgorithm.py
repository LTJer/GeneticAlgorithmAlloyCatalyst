from utils.GeneticOpreators import get_slab_strings
from utils.GeneticOpreators import OperationSelector as OperSelect
from utils.GeneticOpreators import (RandomElementMutation, 
                              RandomElementPermutation,
                              RandommMelting_2Slabs,
                              CutSpliceSlabCrossover,
                              RandomElementPermutation_2Slabs,
                              RandomSurfaceElementMutation)
from utils.GeneralUtils import ML_Pred_Avg_BindingE, ML_Pred_Highest_BindingE
import numpy as np
from ase.ga.population import RankFitnessPopulation
from ase.ga.data import DataConnection
from ase.ga import set_raw_score, get_raw_score
from ase.ga.population import Population
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
import math
import os 
import shutil
import sys 
import argparse
import random
from keras.models import load_model



def TournamentSelection(AllCandidatesPool, SubPoolSize, Nparents = 2):
    
    if SubPoolSize <= 1:
        SubPoolSize = math.ceil(SubPoolSize*len(AllCandidatesPool))
         
    if SubPoolSize == 1:
        SubPoolSize = 2
    
    SubPool = random.sample(AllCandidatesPool, SubPoolSize)
    SubPoolFitness = [float(get_raw_score(i)) for i in SubPool]
    ParentIndices = []
    for i in range(Nparents):
        MaxIndices = np.argmax(SubPoolFitness) 
        SubPoolFitness[MaxIndices] = -np.inf
        ParentIndices.append(MaxIndices)
    Parents = [SubPool[i] for i in ParentIndices]
    
    return Parents, SubPoolSize

    
def get_comp(atoms):
    return atoms.get_chemical_formula()

def noinfo(atoms):
    return 'Alloy'

def duplictionChecker(slabstrings, stringpool):
    
    duplicationBool = []
    chemicalsymbos = stringpool[1]
    nnmats = stringpool[2]
    
    
    if slabstrings[1] in chemicalsymbos:
        duplicationBool.append(True)
    else:
        duplicationBool.append(False)
        
    if slabstrings[2] in nnmats:
        duplicationBool.append(True)
    else:
        duplicationBool.append(False)
    
    return duplicationBool


def PopulationShiftingPlot(InitialDB, FinalDB):
    NPlottingChildrens = 30
    DB_ini = DataConnection(InitialDB)
    popini = Population(data_connection=DB_ini, population_size=NPlottingChildrens)
    popini.update()
    popini = popini.get_current_population()

    DB_iterations = DataConnection(FinalDB)
    popiter = Population(data_connection=DB_iterations, population_size=NPlottingChildrens)
    popiter.update()
    popiter = popiter.get_current_population()

    raw_scorelist_ini = []
    for i in popini:
        raw_scorelist_ini.append(get_raw_score(i))
        
    raw_scorelist_final = []
    for i in popiter:
        raw_scorelist_final.append(get_raw_score(i))

    #plotting 
    fig, ax = plt.subplots(figsize=(6,4), dpi=250)
    ax = sns.kdeplot(data=raw_scorelist_ini, label='Initial Population', ax=ax, fill=True, color = 'gray')
    ax = sns.kdeplot(data=raw_scorelist_final, label='After iterations', ax=ax, fill=True, color ='crimson')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.75)
        
    font_axis_pub = {
            'color':  'black',
            'weight': 'bold',
            'size': 10,
              }

    ax.set_xlabel('Binding Energy of N (eV)',fontdict=font_axis_pub)
    ax.set_ylabel('Probability Density',fontdict=font_axis_pub)
    plt.savefig(f'GA_PopulationShift_{len(raw_scorelist_ini)}_{len(raw_scorelist_final)}.png', dpi=250)

    return


metals = ['Pd', 'Sc', 'Pb', 'Co', 'Ga',
          'Mo', 'Ni', 'Al', 'Mn', 'In',
          'Cu', 'Cr', 'Fe', 'Nb', 'Ru', 
          'Tl', 'Tc', 'Rh', 'Zn', 'Zr', 
          'Pt', 'Ir', 'Sn', 'Ag', 'Hf', 
          'Au', 'Bi', 'Hg', 'Os', 'Cd',
          'V', 'Ta', 'Re', 'Y', 
          'Ti', 'W']



def main(db, DOSkey = './DOSkey_DOS12_N.pickle', MLmodel = "DOSnet_saved.h5",
         EbinSearchMethod = 'high', ParentSelection = 'weel', #high, avg, wheel, tour
         BEcriteria = 3, require_gens = 2,
         SubPoolSize = 0.5, Nparents = 2, traitRatio = 0.1, NtypElement = 4):
    
    print('--------------------Initialize ML Model and Genetic Operators--------------------')
    '''Load DOS net to Predict Avg Binding Energy'''
    with open(f"{DOSkey}", 'rb') as file:
        Dict_metalDOS = pickle.load(file)
    # Define the DOS scaler for ML input rescaling
    Scaler = StandardScaler()
    DOS_key_rescale = np.array([np.hstack((i[0:2000,:9],i[0:2000,:9],i[0:2000,:9],i[0:2000,:9], i[0:2000,:9])) for i in list(Dict_metalDOS.values())]).astype(np.float32)
    RescaleData = DOS_key_rescale.reshape(-1, DOS_key_rescale.shape[2])
    Scaler.fit(RescaleData)
    # Load pre-train DOS net
    model = load_model(MLmodel)
    # Retrieve saved parameters for later checking
    StringPool = db.get_param('StringPool')
    pop_size = db.get_param('population_size') # the number of offspring will be generated during each generation
    # Pass parameters to the population instance
    # A variable_function is required to divide candidates into groups here we use the chemical composition
    pop = RankFitnessPopulation(data_connection=db,
                                population_size=pop_size,
                                variable_function=noinfo)
    pop.update()
    Operatorlist = ([1, 1, 1, 1, 1, 1], 
                    [RandomElementMutation(metals, NtypElement),
                     RandomSurfaceElementMutation(metals, NtypElement),
                     RandomElementPermutation(),
                     RandomElementPermutation_2Slabs(),  
                     RandommMelting_2Slabs(), 
                     CutSpliceSlabCrossover()])
    OperationSelector = OperSelect(*Operatorlist)
    
    VisualizeSamples = []
    uniqueslabs = []
    duplicateslabs = []
    highestBE = 0
    LeftGenerations = require_gens - db.get_generation_number() + 1
    if LeftGenerations <= 0:
        LeftGenerations = 0
        
    # Below is the iterative part of the algorithm
    print(f'Require {LeftGenerations} more generations')
    print(' ')
    print('====================Start Genetic Algorithm====================')
    if LeftGenerations > 0 :
        for generat in range(LeftGenerations):
            if highestBE >= BEcriteria :
                print(f'=======Reach requested Binding Energy requirement : {BEcriteria} eV=======')
                break
            CurrentGeneration = db.get_generation_number()
            new_offsprings= []
            while len(new_offsprings) <= pop_size:
                print(' ')
                print(' ')
                print(f'-----Generation "{CurrentGeneration}", Population size "{len(new_offsprings)}"')
                print(f'-----Highest Binding E: {round(float(highestBE),2)} eV')
                # Select parents based on operator for a new candidate
                AllCandidates = db.get_all_relaxed_candidates()
                
                if ParentSelection == 'wheel':
                    parents = pop.get_two_candidates() 
                    ParentsFormulas = [i.get_chemical_formula(mode='hill', empirical=False) for i in parents]
                    print(f'-----Parents(wheel) {ParentsFormulas}')
                    
                elif ParentSelection == 'tour':
                    parents, subsize = TournamentSelection(AllCandidates, SubPoolSize, Nparents)
                    ParentsFormulas = [i.get_chemical_formula(mode='hill', empirical=False) for i in parents]
                    print(f'-----Parents(tournament) {ParentsFormulas} from sub-pool {subsize} over {len(AllCandidates)}')
                    
                    
                # Select number of traits
                numberatoms = min(len(parents[0]), len(parents[1]))
                Ntraits = math.floor(traitRatio * numberatoms)
                Ntraits = np.random.randint(Ntraits, size=1) + 1
                # Select an operator 
                op, opname = OperationSelector.get_operator()
                offsprings = op.get_new_individual(parents, Ntraits)
                
                # An operator could return None if an offspring cannot be formed by the chosen parents with operator
                BElist = []
                for count, off in enumerate(offsprings):
                    print('---')
                    offname = off.get_chemical_formula(mode='hill', empirical=False)
                    try:
                        NElement = len(set(off.get_chemical_symbols()))
                    except:
                        pass
                    if off is None:
                        print(f'Offspring {count+1} is None')
                    elif  NElement > NtypElement:
                        print(f'Offspring {count+1} {offname} contains {NElement} element > {NtypElement}')
                    else:
                        print(f'Offspring {count+1} {offname} exists with {NElement} elements')
                        #checking duplications
                        off.info['key_value_pairs']['generation'] = CurrentGeneration
                        slabstrings = get_slab_strings(off) # create strings and values for duplication identification
                        indicator = duplictionChecker(slabstrings = slabstrings, stringpool = StringPool)
                       
                        # print(f'Duplicate identification : {indicator}')
                        if int(indicator.count(True)) == 2 :
                            print(f'{offname} is duplicated')
                            duplicateslabs.append(off)
                        else:
                            print(f'{offname} is unique')
                            
                            if EbinSearchMethod == 'high':
                                BE_ML, VisualExample = ML_Pred_Highest_BindingE(model=model, slab = off, 
                                                                            DataRescaler=Scaler, MetalDOS_Dict = Dict_metalDOS,
                                                                            verbose = False)
                                
                            elif EbinSearchMethod == 'avg' :
                                BE_ML, VisualExample = ML_Pred_Avg_BindingE(model=model, slab = off, 
                                                                            DataRescaler=Scaler, MetalDOS_Dict = Dict_metalDOS,
                                                                            verbose = False)
                    
                            set_raw_score(off, float(BE_ML))
                            BElist.append(BE_ML)
                            VisualizeSamples.append(VisualExample)
                            uniqueslabs.append(off)
                            new_offsprings.append(off)
                            StringPool[0].append(slabstrings[0])
                            StringPool[1].append(slabstrings[1])
                            StringPool[2].append(slabstrings[2])
                    
                if BElist != [] :
                    if float(max(BElist)) >= highestBE :
                        highestBE = max(BElist)
                        print(f'Identify better slab with BE : {highestBE}')
            
            
            #calculate average BE over all new offsprings
            AverageGenerationBE = np.mean([float(get_raw_score(newoff)) for newoff in new_offsprings])
            print(f'Generation Average Binding Energy: {AverageGenerationBE} eV')
            #calculate diversity over all new offsprings
            GenerationDiversity = []
            for newoff in new_offsprings:
                GenerationDiversity += newoff.get_chemical_symbols() 
            GenerationDiversity = set(GenerationDiversity)
            print(f'Generation Diversity: {len(GenerationDiversity)} elements')
            print(f'Generation Element: {GenerationDiversity}')
            # add a full relaxed generation at once, this is faster than adding one at a time
            db.add_more_relaxed_candidates(new_offsprings)
            # update the population to allow new candidates to enter
            pop.update()
            
        print(' ')
        print(f'=======Stop genetic algorithm at {generat} generations=======')
        print(f'Generate {len(uniqueslabs)} unique slabs')
        print(f'Generate {len(duplicateslabs)} duplicated slabs')
        print(f'Current database size {len(db.get_all_relaxed_candidates())}')
        print(f'Highset Binding Energy {highestBE} | Au : 2.468 eV')
        PopulationShiftingPlot(InitialDB = iniDB, FinalDB = finalDB)
        
    else:
        print(' ')
        print('=======Reach requested generations, stop genetic algorithm=======')
        print(f'CurrentGenerations : {CurrentGeneration}')
        print(f'LeftGenerations    : {LeftGenerations}')
        print(f'Current database size {len(db.get_all_relaxed_candidates())}')
        PopulationShiftingPlot(InitialDB = iniDB, FinalDB = finalDB)

    return uniqueslabs

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='arguments for training')
    parser.add_argument('--DOSkey', action='store', dest='DOSkey', type=str, required=True, help='DOSs of empty slabs for normalization')
    parser.add_argument('--MLmodel', action='store', dest='MLmodel', type=str, required=True, help='ML model')
    parser.add_argument('--iniDB', action='store', dest='iniDB', type=str, required=True, help='initial DB with empty surfaces-read only')
    parser.add_argument('--finalDB', action='store', dest='finalDB', type=str, required=True, help='output DB')
    parser.add_argument('--Generations', action='store', dest='Generations', type=str, required=True, help='number of generations')
    parser.add_argument('--BEcriteria', action='store', dest='BEcriteria', type=str, required=True, help='stoping binding energy in eV')
    parser.add_argument('--SubPoolSize', action='store', dest='SubPoolSize', type=str, required=True, help='The subpool size for tournament selection')
    parser.add_argument('--Nparents', action='store', dest='Nparents', type=str, required=True, help='number of parents after parent selection')
    parser.add_argument('--NtypElements', action='store', dest='NtypElements', type=str, required=True, help='number of type of elements can be included =the children slabs')
    parser.add_argument('--traitRatio', action='store', dest='traitRatio', type=str, required=True, help='the ratio of overall number of traits could be modified')
    parser.add_argument('--EbinSearchMethod', action='store', dest='EbinSearchMethod', type=str, required=True, help='the ratio of overall number of traits could be modified')
    parser.add_argument('--ParentSelection', action='store', dest='ParentSelection', type=str, required=True, help='the ratio of overall number of traits could be modified')
    args = parser.parse_args()
    
    DOSkey = args.DOSkey
    MLmodel = args.MLmodel
    iniDB = args.iniDB
    finalDB = args.finalDB
    Generations = args.Generations
    BEcriteria = args.BEcriteria
    SubPoolSize = args.SubPoolSize
    Nparents = args.Nparents
    NtypElements = args.NtypElements
    traitRatio = args.traitRatio
    EbinSearchMethod = args.EbinSearchMethod
    ParentSelection  = args.ParentSelection
    
    # Check if dbfile exists
    if os.path.exists(iniDB):
        # Copy dffile to db2file
        shutil.copy(iniDB, finalDB)
        print("Copy 'GA_initial.db' to 'GA_hull.db'")
    else:
        print("'GA_initial.db' does not exist.")
        sys.exit()
        
    db = DataConnection(finalDB)
    
    all_initial_candidates = db.get_all_relaxed_candidates()
    all_initial_metals = db.get_param('metals')
    print('-------------------------Initialize Genetic Algorithm-----------------------------')
    print('Feeding parameters:')
    print(f'                Initial DB size                   : {len(all_initial_candidates)}')
    print(f'                Initial elements                  : {len(all_initial_metals)}')
    print(f'                Request generations               : {Generations}')
    print(f'                Changeable trait ratio            : {traitRatio}')
    print(f'                Number of selection parents       : {Nparents}')
    print(f'                Number of elements in offspring   : {NtypElements}')
    print(f'                Sub Pool Size                     : {SubPoolSize}')
    
    main(db = db, DOSkey = str(DOSkey), MLmodel = str(MLmodel),
                          EbinSearchMethod = str(EbinSearchMethod), ParentSelection = str(ParentSelection), #high, avg, wheel, tour
                            BEcriteria = float(BEcriteria), require_gens = int(Generations),
                            SubPoolSize = float(SubPoolSize), Nparents = int(Nparents), traitRatio = float(traitRatio), NtypElement = int(NtypElements))
