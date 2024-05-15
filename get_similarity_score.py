from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("large-context-llm")

def get_embedding(text, model="text-embedding-3-large"):
    # text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model).data
    response = response[0].embedding
    return response

def store_vector(vectors: list):
    index.upsert(vectors)

def get_unrelated_data(vector: list, topk: int):
    unrelated_data = index.query(vector=vector, top_k=topk)
    unrelated_data_index = int(unrelated_data["matches"][-1]["id"])
    return unrelated_data_index

# def get_similarity_score_batched(queries, documents, batch_size=1):
#     tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
#     model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')
#     max_length = 4096
#
#     all_scores = None
#
#     # Process each document's sentences individually
#     for i in tqdm(range(len(documents))):
#         document_scores = []
#         for sentence in documents[i]:
#             input_texts = queries + [sentence]
#             n = len(queries)
#
#             batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
#             outputs = model(**batch_dict)
#             embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
#
#             # Calculate similarity scores for the current sentence
#             scores = embeddings[:n] @ embeddings[n:].T * 100
#             document_scores.append(scores)
#
#         # Average the scores across all sentences in the document
#         document_scores = torch.mean(torch.stack(document_scores), dim=0)
#
#         # Append document scores to the complete scores matrix
#         if all_scores is None:
#             all_scores = document_scores
#         else:
#             all_scores = torch.cat((all_scores, document_scores), dim=1)
#
#     # Once all documents are processed, find the lowest similarity indices
#     lowest_score_indices = get_lowest_similarity_indices(all_scores.cpu().numpy())
#     return lowest_score_indices
