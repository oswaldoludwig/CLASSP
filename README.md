# CLASSP
 A Biologically-Inspired Approach to **C**ontinual **L**earning through **A**djustment **S**uppression and **S**parsity **P**romotion

This repository hosts the code for CLASSP, a unique continual learning approach inspired by biological learning principles. CLASSP addresses catastrophic forgetting by balancing the acquisition of new information with the preservation of past knowledge. This balance is achieved via two primary mechanisms: a decay rate for weight updates and a threshold on the loss gradient. The decay rate assigns lower learning rates to frequently updated weights, thereby maintaining their relevance to previous tasks. The threshold encourages sparsity, reserving capacity for future tasks. If you use ideas or code from this repository in a publication, please cite our paper DOI: 10.13140/RG.2.2.20942.06724

CLASSP is implemented in a Python/PyTorch class (see CLASSP.py), making it applicable to any model. Usage is like any other optimizer:

from CLASSP import CLASSP_optimizer
optimizer = CLASSP_optimizer(model.parameters(), lr=LR, threshold=Threshold, epsilon=Epsilon, power=Power)

[Read the paper here](https://www.researchgate.net/publication/380184328_CLASSP_a_Biologically-Inspired_Approach_to_Continual_Learning_through_Adjustment_Suppression_and_Sparsity_Promotion)
