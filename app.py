import gradio as gr
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)

model_id = "rasyosef/llama-3.2-amharic-64k-1024"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


def generate(prompt):
    prompt_length = len(tokenizer.tokenize(prompt))
    if prompt_length >= 128:
        yield prompt + "\n\nPrompt is too long. It needs to be less than 128 tokens."
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs["input_ids"][0][0] = 0
        print(inputs)

        max_new_tokens = max(0, 128 - prompt_length)
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            skip_prompt=False,
            skip_special_tokens=True,
            timeout=300.0,
        )
        thread = Thread(
            target=model.generate,
            kwargs={
                "inputs": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": max_new_tokens,
                "temperature": 0.3,
                "do_sample": True,
                "top_k": 8,
                "top_p": 0.8,
                "repetition_penalty": 1.25,
                "streamer": streamer,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            },
        )
        thread.start()

        generated_text = ""
        for word in streamer:
            generated_text += word
            response = generated_text.strip()
            yield response


with gr.Blocks(css="#prompt_textbox textarea {color: blue}") as demo:
    gr.Markdown(
        """
  # Llama 3.2 Amharic
  This is a demo for [llama-3.2-amharic](https://huggingface.co/rasyosef/llama-3.2-amharic-64k-1024), a smaller version of Meta's [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) decoder transformer model pretrained for 3 days on `210 million` tokens of **Amharic** text. This model has `179 million` parameters and a context size of `1024` tokens. This is a base model and hasn't undergone any supervised finetuing yet.
  
  Please **enter a prompt** and click the **Generate** button to generate completions for the prompt.
  #### Text generation parameters:
  - `temperature` : **0.3**
  - `do_sample` : **True**
  - `top_k` : **8**
  - `top_p` : **0.8**
  - `repetition_penalty` : **1.25**
  """
    )

    prompt = gr.Textbox(
        label="Prompt",
        placeholder="Enter prompt here",
        lines=4,
        interactive=True,
        elem_id="prompt_textbox",
    )
    with gr.Row():
        with gr.Column():
            gen = gr.Button("Generate")
        with gr.Column():
            btn = gr.ClearButton([prompt])
    gen.click(generate, inputs=[prompt], outputs=[prompt])
    examples = gr.Examples(
        examples=["አዲስ አበባ", "በእንግሊዙ ፕሬሚየር ሊግ", "ፕሬዚዳንት ዶናልድ ትራምፕ", "በመስቀል አደባባይ"],
        inputs=[prompt],
    )
demo.queue().launch(debug=True)
