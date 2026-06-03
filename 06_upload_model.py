from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model")
model = AutoModelForCausalLM.from_pretrained("checkpoints/pythia_finetuned")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/pythia_finetuned")

model.push_to_hub("pythia-160m-pycode-ft", private=True)
tokenizer.push_to_hub("pythia-160m-pycode-ft", private=True)

print("Done")