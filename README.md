# Critical Reaction time and Short-term memory tests of Centaur 70B

[Centaur](https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B) has been put forth as a "foundation" model of human cognition. It has state-of-art performance on predicting human responses on various psychological tests. In our [commentary](https://osf.io/preprints/psyarxiv/v9w37_v2), we critically assess if it indeed has learnt about fundamental characteristics of human cognition. 

We ran simulations to assess two such fundamental characteristics:
1. Does Centaur implicitly model human reaction times (which are in the hundreds of milliseconds)? By implementing a reaction time report task (test_RT.py), we found that no, Centaur does not implicitly model human RTs—it can respond in 1ms if cued to do so, and it has a very broad spectrum of responses when asked to report freely.
2. Does Centaur implicitly model human short term memory (which is ~4-7 items)? By implementing a digit-recall task (test_STM.py), we found that no, Centaur does not have human-like memory limits—it can perfectly recall many sequences of over a 100 digits!

These experiments cast doubt over the claim that Centaur is modeling human cognition.