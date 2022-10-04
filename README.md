# Quantum-Convolutional-Neural-Network-Stock-Broker
[![forthebadge](https://forthebadge.com/images/badges/built-with-science.svg)](https://forthebadge.com)
[![Gem Version](https://badge.fury.io/rb/colorls.svg)](https://badge.fury.io/rb/colorls)
Based on "Algorithmic financial trading with deep convolutional neural networks: Time series to image conversion approach" and \
"Quanvolutional Neural Networks:Powering Image Recognition with Quantum Circuits"
## Data

*  Google stock close price from 01.01.2015 to 30.09.2022 (yfinance).
* 225 features constructed from 15 technical indicators with 15 different periods each: \
RSI, WillR, FWMA, EMA, SMA, HMA, TripleEMA, CCI, CMO, Inertia, PGO, ROC, CMF, MOM, PSL with periods from 6-20days (pandas-ta).
* Features MinMax scaled and arranged into 15x15 matrix.

## Networks
* Fully classical CNN (Pytorch).
* Classical CNN with one fully connected quantum layer leeding into the output layer (Pytorch + Qiskit).
* CNN where the first convolutional layer is replaced with a quantum convolutional layer based on "Quanvolutional Neural Networks:Powering
Image Recognition with Quantum Circuits"(Pytorch + Qiskit).

## Results

### 500 Epochs on test data (last 456 days).
F1 score (weighted): 0.814\
F1 score (micro): 0.870\
cohen's Kappa: -0.007


* Classical CNN: 
	Loss: 0.3498
	Accuracy: 89.8%
  
* Classical CNN With Quantum Dense Layer:
	Loss: 0.89
	Accuracy: 87.2%

* CNN With Quantum Convolutional Layer:
	Loss: 1.12
	Accuracy: 88.2%
  
  
 ## Sources
 * https://arxiv.org/pdf/1904.04767.pdf 
 * https://www.sciencedirect.com/science/article/abs/pii/S1568494618302151
 * https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html
 * https://www.mdpi.com/2079-9292/11/5/721

 
