# Critical Reaction time and Short-term memory tests of Centaur 70B

[Centaur](https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter) has been put forth as a "foundation" model of human cognition. It has state-of-art performance on predicting human responses on various psychological tests. In our [commentary](https://osf.io/preprints/psyarxiv/v9w37_v2), we critically assess if it indeed has learnt about fundamental characteristics of human cognition. 

We ran simulations to assess two such fundamental characteristics:
1. Does Centaur implicitly model human reaction times (which are in the hundreds of milliseconds)? By implementing a reaction time report task ([test_RT.py](https://github.com/novelmartis/centaur_critical_test/blob/main/test_RT.py)), we found that no, Centaur does not implicitly model human RTs—it can respond in 1ms if cued to do so, and it has a very broad spectrum of responses (~1ms-50s) when asked to report freely.
2. Does Centaur implicitly model human short term memory (which is ~4-7 items)? By implementing a digit-recall task ([test_STM.py](https://github.com/novelmartis/centaur_critical_test/blob/main/test_STM.py)), we found that no, Centaur does not have human-like memory limits—it can perfectly recall many sequences of over a 100 digits!

These experiments cast doubt over the claim that Centaur is modeling human cognition.

### Requirements (what we ran the code with)

- python=3.11
- pytorch=2.5.1
- pytorch-cuda=11.8
- xformers
- unsloth

We ran the experiments on 1 H100. The STM script takes ~6 hours to execute, while the RT script finishes in ~30 minutes.