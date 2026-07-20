# Attention

Attention is mechanism which each token see other token to get information that used to predict next token, Intuively each token query attend to all token key and in the end will given real token value

Q denotes what token have to looking for
K denotes what information that have contains
V denotes real information or actual semantic of token, like "if you want me, then here i am"

## Why Attention Exists?

So, used to be algorithm for Sequential Input like Natural Languange Processing is RNN (Recurrent Neural Network), because RNN have problems towards long-term memory LSTM exists version powerfull than one but the algorithm still have the same issue that's amnesia (the more length token grow previous token become faded slowly) and vanishing gradient every time dealing with long sequence and also this algorithm running sequential which is we can't doing parallel and make training be longer, that's why Attention exists to solve the problem about long-term memory because each each token directly attend to each token and also can running by parallel the model can be scale up to be larger and fit with GPU mechanism nowadays, this is why SOTA LLM use Attention mechanism 

## How To Calculate Attention?  

Oke, since we know essence of attention, we also know how to compute this algorithm.
this is how the way compute Attention

We have matrix Q, K, V each matrix contains token-token
Formula:
```
scores = Q @ K.T / sqrt(dim)
scores = causal_mask(scores)
weight = softmax(scores)
output = weight @ V
```

Before we diving deeper, we must know new things that there's in attention mechanism
- Causal Mask, what causal mask? causal mask is technique masking future token so that token can see the future, we need this in model Autoregressive that have to predict next token
- Softmax, this is activation function that change affinity or scores into probability (range 0 - 1), if Qi @ Ki have high probability then token K means so much for Q or high correlation or semantic alignment or contextual affinity

For example
```
Q = [[2, 3],[4,1]], K.T = [[1,2],[3,4]], V = [[3,2],[5,2]], sqrt(2)
scores = Q @ K.T / sqrt(dim)
scores = [[11, 16],[7,12]] / sqrt(2)
scores = [[7.77817459, 11.31370850], [4.94974747,  8.48528137]]
weight = softmax(scores)
weight = [[0.028318, 0.971682],  [0.028318, 0.971682]]
output = weight @ V
output = [[4.943364, 2.0],[4.943364, 2.0]]
```     
> *Note: not use causal mask, because matrix so small, if want see mechanism of causal see Figure ?*

## Conclusion       
So, Attention is mechanism which token attend to other token to get information that could have predict next token correctly