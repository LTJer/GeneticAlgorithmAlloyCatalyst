#!/bin/bash

#SBATCH --job-name="GeneticAlgorithm"
#SBATCH --tasks=1
#SBATCH --mem=20GB
#SBATCH --export=ALL
#SBATCH --time=5-00:00:00

module load apps/python3/2021.11
conda activate DOSnet

DOSkey="../utils/DOSkey.pickle"
MLmodel="../utils/DOSnetPlusModel.h5"
iniDB="../utils/initialDB_high.db"
finalDB="./GA_hull.db"
EXE="../GeneticAlgorithm.py"
python -u $EXE --DOSkey="$DOSkey" --MLmodel="$MLmodel" --iniDB="$iniDB" --finalDB="$finalDB" --Generations=70 --BEcriteria=10 --SubPoolSize=0.09 --Nparents=2 --NtypElements=5 --traitRatio=0.1 --EbinSearchMethod="high" --ParentSelection="tour"
