# Positional Encoding

Positional Encoding is an architechture transformers that given meaning position for each token

## Why We Need PE?

We know attention mechanism is working at level parallel, because that score between token will the same no matter the position token in sentence e.g. "i love you" == "you love i" the 2 sentence in different position of each token have same meaning if we not add meaning of position (information position), so that's why we need positional encoding

## How We Achieve That?     

Implementation of PE Absolute so easy, we just add vector position to each token that already got embedding appropiate the position token in sentence.   
```python
x   = embedding(x)
pos = torch.arange(x.shape[1])
x   = pos + x
```     

## Is It Absolute PE Have Problem?

Yeah, i think every things have problem, so the problem from Absolute PE is naively add meaning position to meaning token which this make model become longer to convergen because model have to learn the same token in different position and also doesnt explicity capture the relative meaning same token in different position e.g. "I love you" vs "I you love" model have to learn love in position 2 (by indexing) and love position 1 even though it the same token but model see different meaning each token, this is the main problem of Absolute PE, the second one Absolute PE can't handle position that exceeds limits of sequence length model pretrain for simplicity can't extrapolation

How we solve the problem Absolute PE? actually many version PE that solve this problem like Sinusiodal, ALiBi, TJB, Shaw, RoPE. nah in this intuition i will explain deeper for RoPE because (maybe i don't know exactly) this is SOTA architechture.

## What Is RoPE?

RoPE or Rotary Positional Encoding, this is PE make attention dot product only depends on difference token, there's no longer meaning token mixed with meaning position, each token pure meaning or semantic the token itself, by use trigonometry math every token in certain position RoPE rotate as far as position token in sentence * theta, the more near angles between token then dot product that token will high it's meant difference range between that token close to each other, the more far range between token the low dot product between token, but we know we don't want naively let token with far range have low attention, RoPE present term fast column and go slowly faded into last column (theta), so RoPE give tolerance between token that have far range but still have high correlate that's why each column in rope have different theta at every column and also fast and slow column give the token signature position because if only have fast column will occur rotation overlapping or phase overlapping (this is occur by re-entrant angle (periodic)) that make model lost meaning position and the last one RoPE can extrapolation position exceeds limits but only a small number of additional positions    

## Whether RoPE have problem as well?

Yes of course, RoPE "vanilla" have problem that's can't extrapolation position that exceeds the limit too far from limit sequence length model pretrain just extrapolation some additional positions, the more sequence length grow RoPE can't handle it and model become catasthropic, how to solve this? researcher made new variants of RoPE that solve that, consist of:

1. NTK-Aware, this is a Neural Tangent Kernel Aware, NTK streteching wavelength in the way compressed angles each positions use scaling base with scale factor so theta compressed therefore model have tolerance towards token that exceeds limit sequence length (max position) into new limit sequence length (new tolerance) but if position token exceeds new limit sequence length model still inaccurate, so like virtual position additional for some position until new limit sequence length

2. YaRN (Yet Another RoPE extensioN) this is solve problem NTK-Aware naively stretching wavelength and compressed theta each column, because fast column still have the same theta and angles between positions model still differentiate positions each token clearly but NTK-Aware naively stretching without thinking anything middle column affected of stretching model become harder differentiate position token in the middle so make model blurry context between token position that clearly have far range but still have meaning because of range of angle so close to each other and model very difficult to capture semantic meaning in middle paragraph, and the last one YaRN know if we add new max position very long result from softmax become sluggish cause in training model used to max position e.g. 2K but in inference meet 32K token model have to probs each token throughout max sequence and make model confuse because of distribution probs become small and no longer sharpen so YaRN present mscale that immediately to sin and cos so increase magnitudo vector Q and K


## Conclusion   
So, Positional Encoding exist is that model doesn't have meaning position in sequence so we need model know position each token in sequence