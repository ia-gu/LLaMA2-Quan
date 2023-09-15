import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("prompt")
args = parser.parse_args()

import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    args.model, load_in_4bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


prompt = open(args.prompt, "r").read()

# 推論の実行
sequences = pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
print(sequences[0]["generated_text"])