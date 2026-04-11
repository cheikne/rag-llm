# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch


# class LLMModel:
#     def __init__(self, model_name="mistralai/Mistral-7B-Instruct"):
#         # Try loading the specified model, if it fails, load a smaller fallback model
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                 device_map="auto",
#             )
#         except Exception:
#             fallback = "distilgpt2"
#             self.tokenizer = AutoTokenizer.from_pretrained(fallback)
#             self.model = AutoModelForCausalLM.from_pretrained(fallback)

#     def generate(self, prompt, max_tokens=100):
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         if hasattr(self.model, "device"):
#             inputs = inputs.to(self.model.device)
#         outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.2, pad_token_id=self.tokenizer.eos_token_id)
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

class LLMModel:
    def __init__(self, model_name="google/gemma-2-2b-it"):
        print(f"Loading model: {model_name}...")
        
        # Detect Apple Silicon GPU (MPS)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # On Mac M3, bfloat16 is the most efficient format
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, 
            # device_map=self.device, # Map directly to MPS
            # device_map="auto" 
        ).to(self.device) # Move model to the detected device
        print(f"Model loaded on {self.device}")

    def generate(self, prompt, max_tokens=150):
 
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
           # Clear cache before generation to free up Unified Memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens, 
                    do_sample=True, 
                    temperature=0.1, 
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id, # Fix for potential padding issues
                    use_cache=True, # Enable caching for faster generation
                )
                
            # Decode only the newly generated tokens
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the model's answer (remove the prompt part)
            # In Gemma, the output usually contains the prompt text
            # if "model" in full_response:
            #     response = full_response.split("model")[-1].strip()
            # else:
            #     # Fallback: remove the user prompt manually
            #     response = full_response.replace(prompt, "").strip()
                
            return full_response.strip()
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Sorry, I encountered an error while generating the response. Error : {e}"
