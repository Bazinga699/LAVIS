import os

# os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl",device_map="auto")
model = model.cuda()
input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))