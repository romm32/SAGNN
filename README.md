# Wireless Link Scheduling with State-Augmented Graph Neural Networks
This is the repository for the paper of the same name, updated to the code associated to the extended paper titled "Long-Horizon Wireless Link Scheduling with State-Augmented Graph Neural Networks".

### Installation
You can clone the repository as is usually done:

_git clone https://github.com/romm32/SAGNN.git_

We provide a .yml file to set up a conda environment in Ubuntu 22, with which the installation of the packages should become easier.

### Use
The file data_tests is a Jupyter Notebook that enables the generation of a dataset and some analysis of it 
(such as the degree of the nodes, a visual representation of the graphs, etc). You can run the cells in this
file to generate a dataset. After this, you can run the main file inside the conda environment as follows.

_python main.py_

You can also specify training/evaluation parameters as arguments. Some datasets are not available as they hit Github's 100MB limit. You can request them via an email to _rominag@seas.upenn.edu_.

Please cite the papers if you use the code:

````bibtex
@misc{camargo2025wirelesslinkschedulingstateaugmented,
  title={Wireless Link Scheduling with State-Augmented Graph Neural Networks},
  author={Garcia Camargo, Romina and Wang, Zhiyang and NaderiAlizadeh, Navid and Ribeiro, Alejandro},
  year={2025},
  eprint={2505.07598},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  url={https://arxiv.org/abs/2505.07598}
}

The citation for the extended version will be added soon.
