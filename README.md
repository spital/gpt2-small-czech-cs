---
language: cs
widget:
- text: "Umělá inteligence pomůže lidstvu překonat budoucí"
  example_title: "Umělá inteligence ..."
- text: "Současný pokrok v oblasti umělých neuronových sítí představuje"
  example_title: "Současný pokrok ..."
- text: "Z hlediska obecné teorie relativity"
  example_title: "Z hlediska ..."
- text: "Vědci objevili šokující nález - stádo jednorožců žijící v odlehlém, dosud neprobádaném údolí v Andách. Ještě větším překvapením pro vědce byla skutečnost, že jednorožci mluvili"
  example_title: "Vědci objevili ..."
license: cc-by-sa-4.0
tags:
- text-generation
- transformers
- pytorch
- gpt2
datasets: 
- wikipedia
---

# GPT2-small-czech-cs: a Language Model for Czech text generation (and more NLP tasks ...)

## Introduction
GPT2-small-czech-cs is a first experimental model for Czech language based on the GPT-2 small model.

It was trained on Czech Wikipedia using **Transfer Learning and Fine-tuning techniques** in about over a weekend on one GPU NVIDIA GTX 1080ti and with about 1GB of training data (cswiki). A training server with couple GPUs for experiments and one RTX 3080 ti was generously provided by [ONYX engineering, spol. s r.o.](http://www.onyx.cz/).

This experiment is a proof-of-concept that it is possible to get a state-of-the-art language model in any language with low resources.

It was fine-tuned from the [English pre-trained GPT-2 small](https://huggingface.co/gpt2) using the Hugging Face libraries (Transformers and Tokenizers) wrapped into the [fastai2](https://dev.fast.ai/) Deep Learning framework. All the fine-tuning fastai v2 techniques were used. This work was inspired by [Faster than training from scratch — Fine-tuning the English GPT-2 in any language with Hugging Face and fastai v2 (practical case with Portuguese)](https://medium.com/@pierre_guillou/faster-than-training-from-scratch-fine-tuning-the-english-gpt-2-in-any-language-with-hugging-f2ec05c98787), citation below.

It is now available on Hugging Face under [gpt2-small-czech-cs](https://huggingface.co/spital/gpt2-small-czech-cs). We release it under [CC BY SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) (i.e. allowing commercial use).  For further information or requests, please post a Github issue at [Github - gpt2-small-czech-cs](https://github.com/spital/gpt2-small-czech-cs).


## Model description

*Note: information copied/pasted from [Model: gpt2 >> Model description](https://huggingface.co/gpt2#model-description)*

GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences.

More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence, shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the predictions for the token `i` only uses the inputs from `1` to `i` but not the future tokens.

This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks. The model is best at what it was pretrained for however, which is generating texts from a prompt.

## How to use GPT2-small-czech-cs with HuggingFace (PyTorch)

### Load the model and its sub-word tokenizer (Byte-level BPE)

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
pretrained = 'spital/gpt2-small-czech-cs'  # a local directory or huggingface model name
tokenizer = GPT2Tokenizer.from_pretrained(pretrained)
model = GPT2LMHeadModel.from_pretrained(pretrained, pad_token_id=tokenizer.eos_token_id)

# Sequence length max is 1024
tokenizer.model_max_length = 1024

# disable dropout (or leave in train mode to finetune)
model.eval()
```

### Generate one word

```python
import torch
# input sequence
text = "Umělá inteligence pomůže lidstvu překonat budoucí"
inp_tokens = tokenizer(text, return_tensors="pt")

# model output
outputs = model(**inp_tokens, labels=inp_tokens["input_ids"])
loss, logits = outputs[:2]
predicted_index = torch.argmax(logits[0, -1, :]).item()
predicted_text = tokenizer.decode([predicted_index])

# results
print('input text:', text)
print('predicted text:', predicted_text)
# predicted text:  problémy
```

### Generate few full sequences

```python
text = "Umělá inteligence pomůže lidstvu překonat budoucí"
encoded = tokenizer.encode(text, return_tensors='pt')
# torch.random.manual_seed(0)  # if you need reproducibility
sample_outputs = model.generate(encoded, do_sample=True,
             max_length=encoded.size()[1]+20,
             no_repeat_ngram_size=2, top_p=0.95, top_k=50,
             temperature=0.65, num_return_sequences=3)
for i, sample_output in enumerate(sample_outputs): print("{}: {}\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
```

## Limitations and bias

The training data used for this model come from Czech Wikipedia dump. We know it contains a lot of unfiltered content from the internet, which is far from neutral. As the openAI team themselves point out in their model card:

> Because large-scale language models like GPT-2 do not distinguish fact from fiction, we don't support use-cases that require the generated text to be true. Additionally, language models like GPT-2 reflect the biases inherent to the systems they were trained on, so we do not recommend that they be deployed into systems that interact with humans > unless the deployers first carry out a study of biases relevant to the intended use-case. We found no statistically significant difference in gender, race, and religious bias probes between 774M and 1.5B, implying all versions of GPT-2 should be approached with similar levels of caution around use cases that are sensitive to biases around human attributes.

## Author

Czech GPT-2 small was trained and evaluated by [Jiri Spitalsky](https://www.linkedin.com/in/jiri-spitalsky-09400a2) thanks to the computing power of the GPUs and other hardware generously provided by [ONYX engineering, spol. s r.o.](http://www.onyx.cz/).

## Citation
My special thanks go to Pierre Guillou for his work **GPorTuguese-2 (Portuguese GPT-2 small): a Language Model for Portuguese text generation (and more NLP tasks...)**, my work would not be possible without it.
