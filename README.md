# Llama 3.2 Amharic

This is a demo for [llama-3.2-amharic](https://huggingface.co/rasyosef/llama-3.2-amharic-64k-1024), a smaller version of Meta's [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) decoder transformer model pretrained for 3 days on `210 million` tokens of **Amharic** text. This model has `179 million` parameters and a context size of `1024` tokens. This is a base model and hasn't undergone any supervised finetuing yet.

Please **enter a prompt** and click the **Generate** button to generate completions for the prompt.
#### Text generation parameters:
- `temperature` : **0.3**
- `do_sample` : **True**
- `top_k` : **8**
- `top_p` : **0.8**
- `repetition_penalty` : **1.25**

## Demo

The demo has been depolyed to the following HuggingFace space.

https://huggingface.co/spaces/rasyosef/Llama-3.2-Amharic