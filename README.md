# CLASSP
 A Biologically-Inspired Approach to **C**ontinual **L**earning through **A**djustment **S**uppression and **S**parsity **P**romotion

This repository hosts the code for CLASSP, a unique continual learning approach inspired by biological learning principles. CLASSP addresses catastrophic forgetting by balancing the acquisition of new information with the preservation of past knowledge. If you use ideas or code from this repository in a publication, please cite our paper DOI: 10.13140/RG.2.2.20942.06724

CLASSP is implemented in a Python/PyTorch class (see CLASSP.py), making it applicable to any model. Usage is like any other optimizer:

**from CLASSP import CLASSP_optimizer**

**optimizer = CLASSP_optimizer(model.parameters(), lr=LR, threshold=Threshold, epsilon=Epsilon, power=Power)**

See an example of use in the file **experiment_CV.py**

[Read the paper here](https://www.researchgate.net/publication/380184328_CLASSP_a_Biologically-Inspired_Approach_to_Continual_Learning_through_Adjustment_Suppression_and_Sparsity_Promotion)

CLASSP is based on two main principles observed in neuroscience, particularly in the context of synaptic transmission and Long-Term Potentiation (LTP). The first principle is a decay rate over the weight adjustment, which is implemented as a generalization of the AdaGrad optimization algorithm. This means that weights that have received many updates should have lower learning rates as they likely encode important information about previously seen data. However, this principle results in a diffuse distribution of updates throughout the model, as it promotes updates for weights that haven't been previously updated, while a sparse update distribution is preferred to leave weights unassigned for future tasks. Therefore, the second principle introduces a threshold on the loss gradient. This promotes sparse learning by updating a weight only if the loss gradient with respect to that weight is above a certain threshold, i.e. only updating weights with a significant impact on the current loss. Both principles reflect phenomena observed in LTP, where a threshold effect and a gradual saturation of potentiation have been observed. CLASSP is implemented in a Python/PyTorch class, making it applicable to any model. When compared with Elastic Weight Consolidation (EWC) using Computer Vision datasets, CLASSP demonstrates superior performance in terms of accuracy and memory footprint.
