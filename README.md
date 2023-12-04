# SpotLight: Anomaly Detection in Streaming Graphs

## Introduction

This repository contains the implementation of "SPOTLIGHT: Detecting Anomalies in Streaming Graphs," as presented by Dhivya Eswaran et al. in KDD 2018. You can access the full paper [here](https://www.kdd.org/kdd2018/accepted-papers/view/spotlight-detecting-anomalies-in-streaming-graphs).

## Summary of Paper

SpotLight is a randomized sketching-based method designed for detecting sudden changes in dynamic graphs. It is specifically tailored for identifying the appearance and disappearance of dense subgraphs or bi-cliques using sketching techniques. The key problem SpotLight addresses is articulated as follows:

**Problem 1:** Given a stream of weighted, directed/bipartite graphs, $\{G1, G2, . . .\}$, **detect in near real-time** whether $G_t$ contains a sudden (dis)appearance of a large dense directed subgraph using sub-linear memory.

## Implementation

Experiments and the implementation of SpotLight have been carried out using the following resources:

- DARPA dataset: [DARPA Intrusion Detection Evaluation Dataset](https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset)
- Programming Language: Python 3.11.4
- Environment: Jupyter Notebook

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/your-username/spotlight-anomaly-detection.git
cd spotlight-anomaly-detection

2. Install python version 3.11.4 (and above) and necessary packages

3. run python code.
```
