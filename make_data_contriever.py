import json
import os
import random
from make_data import get_document_length_statistics, get_paragraph_num_statistics, get_sampled_texts, check_duplicated

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

def get_sampled_unrel_texts(score_document_list, sample_len, gold_text, rel=True):
    if rel:
        current_len = len(gold_text)
        sampled_texts = [gold_text]
    else:
        current_len = 0
        sampled_texts = []

    for score, document_text in score_document_list:
        for text in document_text:
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
            sampled_rel_texts_8k = get_sampled_unrel_texts(score_document_list, sample_document_len_8k, gold_text)
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
            sampled_rel_texts_4k = get_sampled_unrel_texts(score_document_list, sample_document_len_4k, gold_text)
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

def save_unrel_contriever(rel_data_16k_path, unrel_data_8k_contriever_path, unrel_data_4k_contriever_path):

    with open(rel_data_16k_path, 'r') as rel_16k:
        rel_16k_datasets = json.load(rel_16k)
        document_texts = []
        rel_16k_question_answer_pair_list = []
        unrel_8k_dataset_list = []
        unrel_4k_dataset_list = []
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        model = AutoModel.from_pretrained('facebook/contriever')

        for rel_16k_dataset in tqdm(rel_16k_datasets, desc="saving q-a pair of 16k_rel data ..."):
            question_text = rel_16k_dataset["question_text"]
            gold = rel_16k_dataset["annotations"]["long_answer"]
            question_answer_pair = question_text + "\n" + gold
            document_texts.append(rel_16k_dataset["document_text"])
            rel_16k_question_answer_pair_list.append(question_answer_pair)

        for index, rel_16k_dataset in enumerate(tqdm(rel_16k_datasets, desc="retrieving & saving unrel data ...")):
            score_document_list = []
            document_text = rel_16k_dataset["document_text"]
            question_text = rel_16k_dataset["question_text"]
            gold_text = rel_16k_dataset["annotations"]["long_answer"]
            query_document_sentences = [question_text]
            query_document_sentences.extend(rel_16k_question_answer_pair_list)

            total_document_len = sum(len(paragraph) for paragraph in document_text)
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

            question_unrel_document_list = [question_text]
            question_unrel_document_list.extend(unrel_document)
            inputs = tokenizer(question_unrel_document_list, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

            question_paragraph_score_list = []

            for i in range(len(question_unrel_document_list) - 1):
                score = torch.dot(embeddings[0], embeddings[i + 1]).item()
                question_paragraph_score_list.append((score, document_texts[i]))
            question_paragraph_score_list.sort(reverse=True)

            # 8k_unrel
            # 스코어 리스트를 스코어 오름차순으로 정렬 후 스코어 낮은 paragraphs부터 순차적으로 추출
            score_document_list.sort()

            sampled_rel_texts_8k = get_sampled_unrel_texts(question_paragraph_score_list, sample_document_len_8k, gold_text, rel=False)

            unrel_data_format_8k = {
                "title": rel_16k_dataset["title"],
                "document_text": sampled_rel_texts_8k,
                "question_text": question_text,
                "annotations": rel_16k_dataset["annotations"],
                "document_url": rel_16k_datasets[unrel_document_index]["document_url"],
                "example_id": f"{index}_unrel"
            }
            unrel_8k_dataset_list.append(unrel_data_format_8k)

            # 4k_rel
            sampled_rel_texts_4k = get_sampled_unrel_texts(question_paragraph_score_list, sample_document_len_4k, gold_text, rel=False)

            unrel_data_format_4k = {
                "title": rel_16k_dataset["title"],
                "document_text": sampled_rel_texts_4k,
                "question_text": question_text,
                "annotations": rel_16k_dataset["annotations"],
                "document_url": rel_16k_datasets[unrel_document_index]["document_url"],
                "example_id": f"{index}_unrel"
            }
            unrel_4k_dataset_list.append(unrel_data_format_4k)

        save_json(unrel_data_8k_contriever_path, unrel_8k_dataset_list)
        save_json(unrel_data_4k_contriever_path, unrel_4k_dataset_list)

def save_mixed(rel_data_path, unrel_data_path, mixed_data_path, random_seed=42):
    random.seed(random_seed)
    mixed_data_list = []

    with open(rel_data_path, "r") as rel, open(unrel_data_path, "r") as unrel:
        rel_datasets = json.load(rel)
        unrel_datasets = json.load(unrel)
        total_len = len(rel_datasets)

        for i in range(total_len):
            rel_dataset = rel_datasets[i]
            unrel_dataset = unrel_datasets[i]

            rel_document = rel_dataset["document_text"]
            unrel_document = unrel_dataset["document_text"]

            # Calculate total length of all strings in the list
            total_rel_len = sum(len(text) for text in rel_document)
            total_unrel_len = sum(len(text) for text in unrel_document)

            # Half of the total length
            sample_rel_len = total_rel_len // 2
            sample_unrel_len = total_unrel_len // 2

            # Remove gold document from rel_document
            gold_document = rel_dataset["annotations"]["long_answer"]
            if gold_document in rel_document:
                rel_document.remove(gold_document)

            # Get sampled texts
            sampled_rel = get_sampled_texts(rel_document, sample_rel_len, random_seed)
            sampled_unrel = get_sampled_texts(unrel_document, sample_unrel_len, random_seed)

            related_information = list(range(len(sampled_rel)))

            sampled_rel.extend(sampled_unrel)


            formatted_mixed_data_dict = {
                "title": rel_dataset["title"],
                "document_text": sampled_rel,
                "related_information": related_information,
                "question_text": rel_dataset["question_text"],
                "annotations": rel_dataset["annotations"],
                "document_url": {
                    "rel": rel_dataset["document_url"],
                    "unrel": unrel_dataset["document_url"]
                },
                "example_id": f"{i}_mix"
            }
            mixed_data_list.append(formatted_mixed_data_dict)

        save_json(mixed_data_path, mixed_data_list)


if __name__ == "__main__":
    rel_data_16k_path = "/data/yjoonjang/datasets/long_context_contriever/16k_rel.json"
    unrel_data_16k_path = "/data/yjoonjang/datasets/long_context_contriever/16k_unrel.json"
    mixed_data_16k_path = "/data/yjoonjang/datasets/long_context_contriever/16k_mixed.json"

    rel_data_8k_contriever_path = "/data/yjoonjang/datasets/long_context_contriever/8k_rel.json"
    rel_data_4k_contriever_path = "/data/yjoonjang/datasets/long_context_contriever/4k_rel.json"

    unrel_data_8k_contriever_path = "/data/yjoonjang/datasets/long_context_contriever/8k_unrel.json"
    unrel_data_4k_contriever_path = "/data/yjoonjang/datasets/long_context_contriever/4k_unrel.json"

    mixed_data_8k_contriever_path = "/data/yjoonjang/datasets/long_context_contriever/8k_mix.json"
    mixed_data_4k_contriever_path = "/data/yjoonjang/datasets/long_context_contriever/4k_mix.json"

    # save_rel_contriever(rel_data_16k_path, rel_data_8k_contriever_path, rel_data_4k_contriever_path)
    # save_unrel_contriever(rel_data_16k_path, unrel_data_8k_contriever_path, unrel_data_4k_contriever_path)
    # print("--------")
    # save_mixed(rel_data_8k_contriever_path, unrel_data_8k_contriever_path, mixed_data_8k_contriever_path, random_seed=42)
    # save_mixed(rel_data_4k_contriever_path, unrel_data_4k_contriever_path, mixed_data_4k_contriever_path, random_seed=42)

    base_dir = "/data/yjoonjang/datasets/long_context_contriever"
    for file_name in os.listdir(base_dir):
        print(f"Checking {file_name}...")
        file_path = os.path.join(base_dir, file_name)
        # check_duplicated(file_path)
        # get_paragraph_num_statistics(file_path)
        # get_document_length_statistics(file_path)
        print("\n\n")

    # get_paragraph_num_statistics(rel_data_16k_path)
    # get_paragraph_num_statistics(rel_data_4k_contriever_path)
    # get_document_length_statistics(rel_data_16k_path)
    # get_document_length_statistics(unrel_data_4k_contriever_path)