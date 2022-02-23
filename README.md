# Domain Neural Adaptation (DNA)

This repository contains a paper with supplementary material for the deep domain adaptation approach DNA. A pytorch implementation of the DNA approach will also be included a few months later (due to concerns with some on-going related works).  

In a nutshell, DNA solves the joint distribution mismatch problem in deep domain adaptation for large scale image recognition. To this end, it exploits a Convolutional Neural Network (CNN) to match the source and target joint distributions in the network representation space under the Relative Chi-Square (RCS) divergence. The following figure illustrates this deep joint distribution matching idea.   


![idea](idea.jpg)



For more details of this domain adaptation approach,  please refer to the following IEEE TNNLS work: 

@article{Chen2022Domain,  
  author={Chen, Sentao and Hong, Zijie and Harandi, Mehrtash and Yang, Xiaowei},  
  journal={IEEE Transactions on Neural Networks and Learning Systems},   
  title={Domain Neural Adaptation},   
  year={2022},  
  volume={},  
  number={},  
  pages={1-12},  
  doi={10.1109/TNNLS.2022.3151683}  
  }
