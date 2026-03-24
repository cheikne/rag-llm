from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LLMModel:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct"):
        # Essayer de charger le modèle demandé, sinon fallback sur un modèle public léger
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        except Exception:
            fallback = "distilgpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
            self.model = AutoModelForCausalLM.from_pretrained(fallback)

    def generate(self, prompt, max_tokens=200):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,repetition_penalty=1.2,
        no_repeat_ngram_size=3)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    