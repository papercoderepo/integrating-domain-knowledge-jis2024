# Integrating Domain Knowledge in Multi-Source Classification Tasks

This repository is dedicated to hosting the code for the paper Integrating Domain Knowledge in Multi-Source Classification Tasks, submitted to the Journal on Interactive Systems in 2024.

## Abstract

> This work presents an extended investigation into multi-domain learning techniques within the context of image and audio classification, with a focus on the latter. In machine learning, collections of data obtained or generated under similar conditions are referred to as domains or data sources. However, the distinct acquisition or generation conditions of these data sources are often overlooked, despite their potential to significantly impact model generalization. Multi-domain learning addresses this challenge by seeking effective methods to train models to per- form adequately across all domains seen during the training process. Our study explores a range of model-agnostic multi-domain learning techniques that leverage explicit domain information alongside class labels. Specifically, we delve into three distinct methodologies: a general approach termed Stew, which involves mixing all available data indiscriminately; and two batch domain-regularization methods: Balanced Domains and Loss Sum. These methods are evaluated through several experiments conducted on datasets featuring multiple data sources for audio and im- age classification tasks. Our findings underscore the importance of considering domain-specific information during the training process. We demonstrate that the application of the Loss Sum method yields notable improvements in model performance compared to conventional approaches that blend data from all available domains. By examining the impact of different multi-domain learning techniques on classification tasks, this study contributes to a deeper understanding of effective strategies for leveraging domain knowledge in machine learning model training.

## Contents

The repository contains: 
- Python implementations of the proposed batch domain-regularization techniques;
- Dockerfile and docker-compose.yml to reproduce the development environment used throughout the experiments;
- Jupyter notebooks containing dataset transformations for the creation of training and evaluation sets used in the paper.

## Reproducing experiments:

- Download the datasets
- Spin docker image using the compose command
- Attach to development container and run main.py
- docker-compose down to shutdown the experiments

## Acknowledgments

> We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan X Pascal GPU used for this research. Furthermore, this study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior – Brazil (CAPES) – Finance Code 001.
