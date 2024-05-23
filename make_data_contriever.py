import json
import random

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def save_json(file_name: str, dataset: list):
    with open(file_name, "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

def get_excluded_indices(document_texts, long_answer):
    indices = list(range(0, len(document_texts)))

    # long_answer에 해당하는 인덱스를 제외
    excluded_indices = [index for index in indices if document_texts[index] != long_answer]

    # long_answer에 해당하는 인덱스가 없는 경우
    if len(excluded_indices) == len(indices):
        print("No index found for long_answer !!")
        return

    return excluded_indices

def get_sampled_texts(score_document_list, sample_len, gold_text):
    current_len = len(gold_text)
    sampled_texts = [gold_text]

    for score, text in score_document_list:
        if text != gold_text:
            if current_len + len(text) <= sample_len:
                sampled_texts.append(text)
                current_len += len(text)
            else:
                # If adding the full text exceeds sample_len, truncate the text
                remaining_len = sample_len - current_len
                if remaining_len > 0:
                    sampled_texts.append(text[:remaining_len])
                break

    return sampled_texts
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def retrieve_using_contriever(rel_16k_path, rel_data_8k_contriever_path):
    with open(rel_16k_path, 'r') as rel_16k:
        rel_16k_datasets = json.load(rel_16k)
        rek_8k_dataset_list = []
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        model = AutoModel.from_pretrained('facebook/contriever')

        for index, rel_16k_dataset in enumerate(tqdm(rel_16k_datasets)):
            score_document_list = []
            document_texts = rel_16k_dataset["document_text"]
            question_text = rel_16k_dataset["question_text"]
            gold_text = rel_16k_dataset["annotations"]["long_answer"]
            query_document_sentences = [question_text]
            query_document_sentences.extend(document_texts)

            total_document_len = sum(len(document) for document in document_texts)
            sample_document_len = total_document_len // 2

            inputs = tokenizer(query_document_sentences, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

            for i in range(len(query_document_sentences) - 1):
                score = torch.dot(embeddings[0], embeddings[i + 1]).item()
                score_document_list.append((score, document_texts[i]))

            # 스코어 리스트를 스코어 오름차순으로 정렬
            score_document_list.sort()
            sampled_texts = get_sampled_texts(score_document_list, sample_document_len, gold_text)
            random.seed(42)
            random.shuffle(sampled_texts)

            related_information = get_excluded_indices(sampled_texts, gold_text)

            data_format = {
                "title": rel_16k_dataset["title"],
                "document_text": sampled_texts,
                "related_information": related_information,
                "question_text": question_text,
                "annotations": rel_16k_dataset["annotations"],
                "document_url": rel_16k_dataset["document_url"],
                "example_id": f"{index}_rel"
            }
            rek_8k_dataset_list.append(data_format)
    save_json(rel_data_8k_contriever_path, rek_8k_dataset_list)

if __name__ == "__main__":
    rel_data_16k_path = "/data/yjoonjang/datasets/long_context/16k_rel.json"
    unrel_data_16k_path = "/data/yjoonjang/datasets/long_context/16k_unrel.json"
    mixed_data_16k_path = "/data/yjoonjang/datasets/long_context/16k_mixed.json"

    rel_data_8k_contriever_path = "/data/yjoonjang/datasets/long_context_contriever/8k_rel.json"

    retrieve_using_contriever(rel_data_16k_path, rel_data_8k_contriever_path)