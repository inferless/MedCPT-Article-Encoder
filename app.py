import torch
from transformers import AutoTokenizer, AutoModel


class InferlessPythonModel:
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self.model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder",device_map="cuda")

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
