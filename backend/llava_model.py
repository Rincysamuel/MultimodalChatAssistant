# backend/llava_model.py

import torch
from llava.model.builder import load_pretrained_model  # ✅ correct import

class LLaVAWrapper:
    def __init__(self, device='cpu'):
        self.device = device
        print(f"Device set to use {device}")

        # ✅ load_pretrained_model now needs 3 args + device
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path="liuhaotian/llava-v1.6-mistral-7b",  # or any other model
            model_base=None,
            model_name="llava-v1.6-mistral-7b",
            device=device
        )

        self.model.eval()

    @torch.no_grad()
    def predict(self, text_inputs, image_inputs=None):
        """
        text_inputs: list of str
        image_inputs: list of PIL Images (optional)
        """
        outputs = []
        for text in text_inputs:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            # Forward pass
            out = self.model.generate(**inputs)
            outputs.append(self.tokenizer.decode(out[0], skip_special_tokens=True))
        return outputs
