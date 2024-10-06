
# Mistral-7B-Instruct Model for Text Generation

This repository demonstrates how to use the **Mistral-7B-Instruct-v0.1-GPTQ** model for text generation tasks. The code loads the pre-trained model, defines utility functions for generating text predictions, and provides two methods to generate text using the model.

## Requirements

To execute the code, install the following Python packages:

```bash
pip install transformers torch
```

## Model Configuration

The model used in this project is **TheBloke/Mistral-7B-Instruct-v0.1-GPTQ**, a causal language model designed for generating human-like text responses.

```python
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
```

The prompt used for text generation is:

```python
prompt = "Tell me about AI"
prompt_template=f'''<s>[INST] {prompt} [/INST]
'''
```

## Utility Methods

Two utility functions are provided for model loading and text generation:

1. **get_model**: This function loads the pre-trained model and its tokenizer.
2. **get_prediction**: Generates a text prediction using the model by directly feeding input IDs into the model.
3. **get_prediction2**: Generates text using the `pipeline` method from the `transformers` library.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def get_model():
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=False, revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model, tokenizer

def get_prediction(model, tokenizer):
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    return tokenizer.decode(output[0])

def get_prediction2(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1)
    return pipe(prompt_template)[0]['generated_text']
```

## Execution

Two methods are provided for text generation:

### Method 1: Direct Model Generation

This method uses the `generate` function of the model to produce text.

```python
import time

start = time.time()
predicted_text = get_prediction(model, tokenizer)
end = time.time()

# Display the prediction and execution time
print(f"Predicted Text: {predicted_text}")
print(f"Time taken: {(end-start)*10**3:.03f}ms")
```

### Method 2: Pipeline Text Generation

This method leverages the `pipeline` utility for text generation.

```python
import time

start = time.time()
predicted_text = get_prediction2(model, tokenizer)
end = time.time()

# Display the prediction and execution time
print(f"Predicted Text: {predicted_text}")
print(f"Time taken: {(end-start)*10**3:.03f}ms")
```

## Conclusion

This script demonstrates two ways of generating text using the Mistral-7B-Instruct model. Both methods are effective, but the choice between them may depend on specific use cases and performance requirements.
