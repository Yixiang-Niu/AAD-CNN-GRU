# AAD-CNN-GRU
Python code for "Music-oriented auditory attention detection from electroencephalogram", submitted to _Neuroscience Letters_
## Overview
Two separate paths are adopted to generate the EEG embedding and sound source embedding, respectively. Each path is composed of a dilated one-dimensional convolution layer, a batch normalization layer, and a gated recurrent unit layer. The attended sound source is supposed to be more similar to the EEG than the unattended one in the embedding space.
## Files
**model-early.py:** an early version of the model developed for the task one of the ICASSP 2023 “Auditory EEG Decoding” Signal Processing Grand Challenge

**model.py:** the modified version of the model reported in "Music-oriented auditory attention detection from electroencephalogram" (submitted)
