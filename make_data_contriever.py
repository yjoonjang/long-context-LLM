import json
import random
from make_data import get_document_length_statistics, get_paragraph_num_statistics

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def save_json(file_name: str, dataset: list):
    with open(file_name, "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

def get_excluded_indices(document_texts, gold):
    # gold를 가운데에 위치시키기
    if gold not in document_texts:
        print("Gold value not found in the document texts!")
        return

    document_texts.remove(gold)
    middle_index = len(document_texts) // 2
    document_texts.insert(middle_index, gold)

    # gold의 인덱스를 찾기
    gold_index = document_texts.index(gold)

    # excluded_indices 계산
    excluded_indices = [i for i in range(len(document_texts)) if i != gold_index]

    return excluded_indices

def get_sampled_texts(score_document_list, sample_len, gold_text, rel=True):
    if rel:
        current_len = len(gold_text)
        sampled_texts = [gold_text]
    else:
        current_len = 0
        sampled_texts = []

    for score, text in score_document_list:
        if text != gold_text:
            if (current_len + len(text) <= sample_len) or (len(sampled_texts) < 3):
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

def save_rel_contriever(rel_data_16k_path, rel_data_8k_contriever_path, rel_data_4k_contriever_path):

    with open(rel_data_16k_path, 'r') as rel_16k:
        rel_16k_datasets = json.load(rel_16k)
        rel_8k_dataset_list = []
        rel_4k_dataset_list = []
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
            sample_document_len_8k = total_document_len // 2
            sample_document_len_4k = total_document_len // 4

            inputs = tokenizer(query_document_sentences, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

            for i in range(len(query_document_sentences) - 1):
                score = torch.dot(embeddings[0], embeddings[i + 1]).item()
                score_document_list.append((score, document_texts[i]))

            # 8k_rel
            # 스코어 리스트를 스코어 내림차순으로 정렬 후 스코어 높은 paragraphs부터 순차적으로 추출
            score_document_list.sort(reverse=True)
            sampled_rel_texts_8k = get_sampled_texts(score_document_list, sample_document_len_8k, gold_text)
            related_information_8k = get_excluded_indices(sampled_rel_texts_8k, gold_text)

            rel_data_format_8k = {
                "title": rel_16k_dataset["title"],
                "document_text": sampled_rel_texts_8k,
                "related_information": related_information_8k,
                "question_text": question_text,
                "annotations": rel_16k_dataset["annotations"],
                "document_url": rel_16k_dataset["document_url"],
                "example_id": f"{index}_rel"
            }
            rel_8k_dataset_list.append(rel_data_format_8k)

            # 4k_rel
            sampled_rel_texts_4k = get_sampled_texts(score_document_list, sample_document_len_4k, gold_text)
            related_information_4k = get_excluded_indices(sampled_rel_texts_4k, gold_text)

            rel_data_format_4k = {
                "title": rel_16k_dataset["title"],
                "document_text": sampled_rel_texts_4k,
                "related_information": related_information_4k,
                "question_text": question_text,
                "annotations": rel_16k_dataset["annotations"],
                "document_url": rel_16k_dataset["document_url"],
                "example_id": f"{index}_rel"
            }
            rel_4k_dataset_list.append(rel_data_format_4k)

        save_json(rel_data_8k_contriever_path, rel_8k_dataset_list)
        save_json(rel_data_4k_contriever_path, rel_4k_dataset_list)

def save_unrel_contriever(rel_data_16k_path, rel_data_8k_contriever_path, rel_data_4k_contriever_path):

    with open(rel_data_16k_path, 'r') as rel_16k:
        rel_16k_datasets = json.load(rel_16k)
        rel_16k_question_answer_pair_list = []
        rel_8k_dataset_list = []
        rel_4k_dataset_list = []
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        model = AutoModel.from_pretrained('facebook/contriever')

        for rel_16k_dataset in tqdm(rel_16k_datasets, desc="saving q-a pair of 16k_rel data ..."):
            question_text = rel_16k_dataset["question_text"]
            gold = rel_16k_dataset["annotations"]["long_answer"]
            question_answer_pair = question_text + "\n" + gold
            rel_16k_question_answer_pair_list.append(question_answer_pair)

        for index, rel_16k_dataset in enumerate(tqdm(rel_16k_datasets, desc="retrieving & saving unrel data ...")):
            score_document_list = []
            document_texts = rel_16k_dataset["document_text"]
            question_text = rel_16k_dataset["question_text"]
            gold_text = rel_16k_dataset["annotations"]["long_answer"]
            query_document_sentences = [question_text]
            query_document_sentences.extend(rel_16k_question_answer_pair_list)

            total_document_len = sum(len(document) for document in document_texts)
            sample_document_len_8k = total_document_len // 2
            sample_document_len_4k = total_document_len // 4

            inputs = tokenizer(query_document_sentences, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

            for i in range(len(query_document_sentences) - 1):
                score = torch.dot(embeddings[0], embeddings[i + 1]).item()
                score_document_list.append((score, document_texts[i]))

            min_score, unrel_document = min(score_document_list, key=lambda x: x[0])
            unrel_document_index = score_document_list.index((min_score, unrel_document))

            question_unrel_document_list = [question_text].extend(unrel_document)
            inputs = tokenizer(question_unrel_document_list, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

            for i in range(len(question_unrel_document_list) - 1):
                score = torch.dot(embeddings[0], embeddings[i + 1]).item()
                score_document_list.append((score, document_texts[i]))
            score_document_list.sort(reverse=True)


            # 8k_unrel
            # 스코어 리스트를 스코어 오름차순으로 정렬 후 스코어 낮은 paragraphs부터 순차적으로 추출
            score_document_list.sort()

            sampled_rel_texts_8k = get_sampled_texts(score_document_list, sample_document_len_8k, gold_text, rel=False)

            rel_data_format_8k = {
                "title": rel_16k_dataset["title"],
                "document_text": sampled_rel_texts_8k,
                "question_text": question_text,
                "annotations": rel_16k_dataset["annotations"],
                "document_url": rel_16k_datasets[unrel_document_index]["document_url"],
                "example_id": f"{index}_rel"
            }
            rel_8k_dataset_list.append(rel_data_format_8k)

            # 4k_rel
            sampled_rel_texts_4k = get_sampled_texts(score_document_list, sample_document_len_4k, gold_text, rel=False)

            rel_data_format_4k = {
                "title": rel_16k_dataset["title"],
                "document_text": sampled_rel_texts_4k,
                "question_text": question_text,
                "annotations": rel_16k_dataset["annotations"],
                "document_url": rel_16k_datasets[unrel_document_index]["document_url"],
                "example_id": f"{index}_rel"
            }
            rel_4k_dataset_list.append(rel_data_format_4k)





if __name__ == "__main__":
    rel_data_16k_path = "/data/yjoonjang/datasets/long_context_contriever/16k_rel.json"
    unrel_data_16k_path = "/data/yjoonjang/datasets/long_context_contriever/16k_unrel.json"
    mixed_data_16k_path = "/data/yjoonjang/datasets/long_context_contriever/16k_mixed.json"

    rel_data_8k_contriever_path = "/data/yjoonjang/datasets/long_context_contriever/8k_rel.json"
    rel_data_4k_contriever_path = "/data/yjoonjang/datasets/long_context_contriever/4k_rel.json"

    save_rel_contriever(rel_data_16k_path, rel_data_8k_contriever_path, rel_data_4k_contriever_path)
    # print("--------")
    # get_paragraph_num_statistics(rel_data_8k_contriever_path)
    # get_paragraph_num_statistics(rel_data_4k_contriever_path)