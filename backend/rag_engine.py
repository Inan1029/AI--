import torch
from sentence_transformers import SentenceTransformer, util

class RAGEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, documents):
        return self.model.encode(documents, convert_to_tensor=True)

    def encode_query(self, query):
        return self.model.encode(query, convert_to_tensor=True)

    def search(self, query, documents, top_k=5):
        query_embedding = self.encode_query(query)
        document_embeddings = self.encode_documents(documents)

        cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        return top_results.indices.tolist(), top_results.values.tolist()
