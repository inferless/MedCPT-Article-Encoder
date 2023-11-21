import torch
from transformers import AutoTokenizer, AutoModel


class InferlessPythonModel:
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self.model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")

    def infer(self, inputs):
        articles = inputs["articles"]
        embeds = None
        with torch.no_grad():
            encoded = self.tokenizer(
                articles,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )

            # encode the queries (use the [CLS] last hidden states as the representations)
            embeds = self.model(**encoded).last_hidden_state[:, 0, :]

        return {"embeds": embeds.tolist()}

    def finalize(self):
        self.tokenizer = None
        self.model = None
