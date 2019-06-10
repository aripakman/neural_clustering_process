# Neural Clustering Process
Implementation of the Neural Clustering Process (NCP) algorithm.
Based on the papers:

- Ari Pakman and Liam Paninski, , [Amortized Bayesian Inference for Clustering Models](https://arxiv.org/abs/1811.09747), BNP@NeurIPS 2018 Workshop

- Ari Pakman, Yueqi Wang, Catalin Mitelut, JinHyung Lee and Liam Paninski, [Discrete Neural Processes](https://arxiv.org/abs/1901.00409), arXiv:1901.00409

```bash
pip3 install -r requirements.txt
python3 main.py --model Gauss2D
python3 main.py --model MNIST
```

The code contains two implementations of the NCP algorithm, which differ in the way GPU parallelism is handled:
1. In ```ncp.py,``` used at train time, parallelization is over a minibatch of datasets, all with the same size and cluster structure.
2. In ```ncp_sampler.py,``` used at test time, only one dataset is used, and samples with different cluster structures are generated in parallel.



