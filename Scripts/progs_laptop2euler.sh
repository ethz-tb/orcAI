#!/bin/bash
#rsync -avz --include='*.py' --include='*.dict' --include='*.list' --include='*.csv' --exclude='*' /Users/sb/polybox/Documents/Research/Sebastian/OrcAI_project/ sbonhoef@euler.ethz.ch:OrcAI_project/
rsync -avz Train_and_Test/*.py sbonhoef@euler.ethz.ch:OrcAI_project/Train_and_Test/
rsync -avz MakeData/*.py sbonhoef@euler.ethz.ch:OrcAI_project/MakeData/
rsync -avz GenericParameters/* sbonhoef@euler.ethz.ch:OrcAI_project/GenericParameters/
rsync -avz Scripts/*.sh sbonhoef@euler.ethz.ch:OrcAI_project/Scripts/
rsync -avz *.py sbonhoef@euler.ethz.ch:OrcAI_project/

