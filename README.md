# Transport Score Climbing comparative analysis 
DL Skoltech Project
\
Authors: Elfat Sabitov, Varvara Furik, Boris Mikheev, Stanislav Efimov, Ekaterina Filimoshina

## Abstract
This project is devoted to MCMC-based sampling methods. These methods are one of the most popular ways of solving a problem of sampling from complicated distributions in high dimensions. Recent MCMC sampling research is deeply connected with the concept of Normalizing Flows -- functions, parameterized with neural networks, which are trained to transform simple distributions into complicated target distribution. One of the approaches within this research area is called Markovian score climbing (MSC) and is presented in [1]. Other important research area is Warped space sampling -- sampling on space obtained with a trainable mapping, which is supposedly simpler to work with. One of the approaches within this area is called Transport score climbing (TSC) and is presented in [2]. Our project aims to compare these approaches with each other and with the classical MCMC methods (MALA, iSIR, and HMC) to test their efficiency in higher dimensional problems. The approaches are compared in different settings in terms of classical metrics for MCMC algorithms such as: Empirical sliced total variation distance (ESTV), Effective sample size (ESS), and Earth mover's distance (EMD) [3]. 

## Our Contribution and Results
Our contribution in the research consists in providing experiments on MSC and TSC in the high-dimensional setting, analyzing the results and comparing methods with each other and classical MCMC methods. 
We implemented:
* Training pipelines for MSC and TSC,
* Banana and Funnel distributions classes,
* Functions for metrics (EMD, ESS, ESTV) calculation,

and conducted experiments on sampling from Banana and Funnel distributions by MSC, TSC, HMC, iSIR, and NUTS for dim = 2, 10, 20, 50, 100.



## Code Organization
* `distributions.py`: Contains code for generating and handling synthetic distributions (funnel and banana).
* `samplers.py`: Contains code for basic MCMC sampling methods (HMC, iSIR, NUTS).
* `metrics.py`: Contains code for ESS, EMD, ESTV calculation and visualization.
* `experiments/`: Contains notebooks with MSC and TSC (with Affine transform and RealNVP) training loops and experiments:
  * `True_Banana_Funnel.ipynb`: Check of Banana and Funnel distributions classes.
  * `TSC_RealNVP_and_Experiments.ipynb`: TSC with RealNVP training loop and experiments on Banana and Funnel in dim=2, 10, 50, 100.
  * `MSC_and_Affine_TSC.ipynb`: MSC training loop and experiments on Banana and Funnel.
  * `TSC_Affine.ipynb`: TSC with Affine transformation training loop and experiments on Banana and Funnel.


## References

[1] C. A. Naesseth, F. Lindsten, D. Blei: Markovian score climbing: Variational Inference with KL. https://www.cs.columbia.edu/~blei/papers/NaessethLindstenBlei2020.pdf

[2] L. Zhang, D. M. Blei, C. A. Naesseth: Transport score climbing: Variational Inference Using Forward KL and Adaprive Neural Transport. https://arxiv.org/abs/2202.01841

[3] S. Samsonov, E. Lagutin, M. Gabrie, A. Durmus, A. Naumov, E. Moulines: Local-Global MCMC kernels: the best of both worlds. https://proceedings.neurips.cc/paper_files/paper/2022/file/21c86d5b10cdc28664ccdadf0a29065a-Paper-Conference.pdf



