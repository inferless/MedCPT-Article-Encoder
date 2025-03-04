import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer, AutoModel


class InferlessPythonModel:
    def initialize(self):
        model_id = "ncbi/MedCPT-Article-Encoder"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id,device_map="cuda")

    def infer(self, inputs):
        articles = inputs["articles"]
        nested_articles = [articles[i:i + 2] for i in range(0, len(articles), 2)]
        embeds = None
        with torch.no_grad():
            encoded = self.tokenizer(
                nested_articles,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            ).to("cuda")

            embeds = self.model(**encoded).last_hidden_state[:, 0, :]

        return {"embeds": embeds.tolist()}

    def finalize(self):
        self.model = None
