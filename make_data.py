from datasets import load_dataset
from tqdm import tqdm
from html.parser import HTMLParser
import json
import jsonlines
import random
import matplotlib.pyplot as plt
from get_similarity_score import get_embedding, store_vector, get_unrelated_data
import torch
import torch.nn as nn


class TextFromPParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.data = []
        self.p_tag_open = False
        self.inside_table = False  # <table> 태그 내부에 있는지 확인하기 위한 플래그

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'table':
            self.inside_table = True  # <table> 태그 시작
        if tag.lower() == 'p' and not self.inside_table:
            self.recording = True
            self.p_tag_open = True

    def handle_endtag(self, tag):
        if tag.lower() == 'table':
            self.inside_table = False  # <table> 태그 종료
        if tag.lower() == 'p':
            self.recording = False
            self.p_tag_open = False

    def handle_data(self, data):
        if self.recording and self.p_tag_open and not self.inside_table:
            self.data.append(data)

def extract_text_from_p(html_content):
    parser = TextFromPParser()
    parser.feed(html_content)
    return parser.data

def extract_title(text):
    hyphen_position = text.find(' - wikipedia')
    if hyphen_position != -1:
        title = text[:hyphen_position]
    else:
        hyphen_position = text.find(' - Wikipedia')
        title = text[:hyphen_position]
    return title


count = 0
def get_excluded_indices(document_texts, long_answer):
    indices = list(range(0, len(document_texts)))

    # long_answer에 해당하는 인덱스를 제외
    excluded_indices = [index for index in indices if document_texts[index] != long_answer]

    # long_answer에 해당하는 인덱스가 없는 경우
    if len(excluded_indices) == len(indices):
        print("No index found for long_answer !!")
        return

    return excluded_indices


def save_json(file_name, dataset):
    with open(file_name, "w") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)

def get_document_length_statistics(file_name):
    with open(file_name, "r") as file:
        datasets = json.load(file)
        document_text_lengths = []

        for dataset in datasets:
            total_document_text = ''
            document_texts = dataset["document_text"]
            for document_text in document_texts:
                total_document_text += document_text
                total_document_text += "\n"
            document_text_lengths.append(len(total_document_text))
        print(f"MAX: {max(document_text_lengths)}")
        print(f"MIN: {min(document_text_lengths)}")
        print(f"AVG: {sum(document_text_lengths) / len(document_text_lengths)}")

def save_rel(input_file, output_file):
    with open(input_file, "r") as file:
        formatted_datasets = []
        datasets = json.load(file)

        # random sampling 할거면 활성화
        # random.seed(42)
        # random_datasets = random.sample(datasets, 100)

        for index, dataset in enumerate(datasets):
            document_text = dataset["document_text"]

            # check if 15.9K < len(document_text) <= 16K & exists (long_answer & short_answers) & "yes_no_answer": "NONE"
            # annotations = dataset["annotations"]
            # if ((15900 <= len(document_text)) and (len(document_text) <= 16000)) and (annotations[0]["long_answer"]) and (len(annotations[0]["short_answers"]) >= 1) and (annotations[0]["yes_no_answer"] == "NONE"):

            title = extract_title(document_text)
            question_text = dataset["question_text"]
            extracted_document_texts = extract_text_from_p(document_text)

            # filter
            banned_text_list = [" ", " . ", "<", "  "]
            extracted_document_texts = [text for text in extracted_document_texts if (text not in banned_text_list) and (len(text) >= 30)] # filter unimportant document_text
            extracted_document_texts = list(set(extracted_document_texts)) # deduplicate

            document_text_tokens = document_text.split()  # tokenize document_text

            long_answer_info = dataset["annotations"][0]["long_answer"]
            short_answers_info = dataset["annotations"][0]["short_answers"]

            short_answers = []
            long_answer = " ".join(document_text_tokens[long_answer_info['start_token']:long_answer_info['end_token']])
            long_answer = extract_text_from_p(long_answer)[0] # delete P tag from long_answer
            for short_answer_info in short_answers_info:
                short_answer = document_text_tokens[short_answer_info['start_token']:short_answer_info['end_token']]
                short_answers.append(" ".join(short_answer))

            indices_list = get_excluded_indices(extracted_document_texts, long_answer)

            formatted_data_dict = {
                "title": title,
                "document_text": extracted_document_texts,
                "related_information": indices_list,
                "question_text": question_text,
                "annotations": {
                    "yes_no_answer": "NONE",
                    "long_answer": long_answer,
                    "short_answers": short_answers
                },
                "document_url": dataset["document_url"],
                "example_id": f"{index}_rel"
            }
            formatted_datasets.append(formatted_data_dict)
    save_json(output_file, formatted_datasets)

def sample_data(input_file, output_file):
    with open(input_file, "r") as file:
        formatted_datasets = []
        datasets = json.load(file)

        random.seed(42)
        random_datasets = random.sample(datasets, 100)

        for index, dataset in enumerate(random_datasets):
            formatted_data_dict = {
                "title": dataset["title"],
                "document_text": dataset["document_text"],
                "related_information": dataset["related_information"],
                "question_text": dataset["question_text"],
                "annotations": {
                    "yes_no_answer": "NONE",
                    "long_answer": dataset["annotations"]["long_answer"],
                    "short_answers": dataset["annotations"]["short_answers"],
                },
                "document_url": dataset["document_url"],
                "example_id": f"{index}_rel"
            }
            formatted_datasets.append(formatted_data_dict)
        save_json(output_file, formatted_datasets)


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def format_to_vector_dict(vector_id: str, values: list, metadata: dict):
    return {
        "id": vector_id,
        "values": values,
        "metadata": metadata
    }

def store_vector_to_pinecone(document_texts):
    vectors = []
    for i in range(len(document_texts)):
        values = get_embedding(document_texts[i])
        vector_id = f"{i}" # id는 반드시 str이어야 함
        metadata = {
            "rel_document_text_index": i
        }
        vector_dict = format_to_vector_dict(vector_id, values, metadata)
        vectors.append(vector_dict)
    store_vector(vectors)


def save_unrel(rel_data_path, unrel_data_path):
    with open(rel_data_path, 'r') as rel:
        rel_datas = json.load(rel)
        unrel_datas = []
        topk=100

        for index, rel_data in tqdm(enumerate(rel_datas)):
            question_text = rel_data["question_text"]
            # retrieve unsimilar document
            question_text_embedding = get_embedding(question_text)
            unrelated_data_index = get_unrelated_data(question_text_embedding, topk=topk)
            unrelated_document = rel_datas[unrelated_data_index]["document_text"]
            unrelated_document_url = rel_datas[unrelated_data_index]["document_url"]
            unrel_data_dict = {
                "title": rel_data["title"],
                "document_text": unrelated_document,
                "question_text": question_text,
                "annotations": rel_data["annotations"],
                "document_url": unrelated_document_url,
                "example_id": f"{index}_unrel"
            }
            unrel_datas.append(unrel_data_dict)

        save_json(unrel_data_path, unrel_datas)


def get_sampled_texts(document_texts, sample_len, random_seed=42):
    random.seed(random_seed)
    random.shuffle(document_texts)
    current_len = 0
    sampled_texts = []

    for text in document_texts:
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

# rel data에서 반, unrel data에서 반 sampling
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
            sampled_rel.extend(sampled_unrel)

            related_information = list(range(len(sampled_rel)))

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

def get_8k_4k_dataset(dataset, cos, is_rel=True):
    processed_data_8k = []
    processed_data_4k = []

    for index, data in tqdm(enumerate(dataset)):
        gold_text = data["annotations"]["long_answer"]
        question_text = data["question_text"]
        question_embedding = torch.tensor(get_embedding(question_text)).unsqueeze(0)

        document_texts = data["document_text"]
        document_text_embeddings = []

        paragraph_len = sum(len(text) for text in document_texts)
        for document_text in document_texts:
            document_text_embedding = torch.tensor(get_embedding(document_text))
            document_text_embeddings.append(document_text_embedding)

        document_text_embeddings = torch.stack(document_text_embeddings)

        with torch.no_grad():
            similarity = cos(question_embedding, document_text_embeddings)

        sim_len_pairs = list(zip(similarity.tolist(), document_texts))

        if is_rel:
            sim_len_pairs.sort(key=lambda x: x[0], reverse=True)
        else:
            sim_len_pairs.sort(key=lambda x: x[0])

        selected_texts_8k = [gold_text] if is_rel else []
        selected_texts_4k = [gold_text] if is_rel else []

        selected_len_8k = len(gold_text) if is_rel else 0
        selected_len_4k = len(gold_text) if is_rel else 0
        half_paragraph_len = paragraph_len * 0.5
        quarter_paragraph_len = paragraph_len * 0.25

        for sim, text in sim_len_pairs:
            if text != gold_text:
                text_len = len(text)
                if selected_len_8k + text_len <= half_paragraph_len:
                    selected_texts_8k.append(text)
                    selected_len_8k += text_len
                else:
                    remaining_len = half_paragraph_len - selected_len_8k
                    if remaining_len > 0:
                        selected_texts_8k.append(text[:int(remaining_len)])
                    break

        for sim, text in sim_len_pairs:
            if text != gold_text:
                text_len = len(text)
                if selected_len_4k + text_len <= quarter_paragraph_len:
                    selected_texts_4k.append(text)
                    selected_len_4k += text_len
                else:
                    remaining_len = quarter_paragraph_len - selected_len_4k
                    if remaining_len > 0:
                        selected_texts_4k.append(text[:int(remaining_len)])
                    break

        random.shuffle(selected_texts_8k)
        random.shuffle(selected_texts_4k)

        dataset_dict_8k = {
            "title": data["title"],
            "document_text": selected_texts_8k,
            "question_text": data["question_text"],
            "annotations": data["annotations"],
            "document_url": data["document_url"],
            "example_id": f"{index}_{'rel' if is_rel else 'unrel'}"
        }

        dataset_dict_4k = {
            "title": data["title"],
            "document_text": selected_texts_4k,
            "question_text": data["question_text"],
            "annotations": data["annotations"],
            "document_url": data["document_url"],
            "example_id": f"{index}_{'rel' if is_rel else 'unrel'}"
        }

        if is_rel:
            indices_list_8k = get_excluded_indices(selected_texts_8k, gold_text)
            indices_list_4k = get_excluded_indices(selected_texts_4k, gold_text)
            dataset_dict_8k["related_information"] = indices_list_8k
            dataset_dict_4k["related_information"] = indices_list_4k

        processed_data_8k.append(dataset_dict_8k)
        processed_data_4k.append(dataset_dict_4k)

    return processed_data_8k, processed_data_4k


def save_8k_4k(
        rel_data_16k_path,
        unrel_data_16k_path,
        rel_data_8k_path,
        unrel_data_8k_path,
        rel_data_4k_path,
        unrel_data_4k_path,
        random_seed=42
    ):
    random.seed(random_seed)
    cos = nn.CosineSimilarity(dim=-1)

    with open(rel_data_16k_path, 'r') as rel_16k, open(unrel_data_16k_path, 'r') as unrel_16k:
        rel_16k_datasets = json.load(rel_16k)
        unrel_16k_datasets = json.load(unrel_16k)

        rel_datasets_8k, rel_datasets_4k = get_8k_4k_dataset(rel_16k_datasets, cos, is_rel=True)
        unrel_datasets_8k, unrel_datasets_4k = get_8k_4k_dataset(unrel_16k_datasets, cos, is_rel=False)

        save_json(rel_data_8k_path, rel_datasets_8k)
        save_json(rel_data_4k_path, rel_datasets_4k)
        save_json(unrel_data_8k_path, unrel_datasets_8k)
        save_json(unrel_data_4k_path, unrel_datasets_4k)


if __name__ == "__main__":
    source_file = "/data/koo/datasets/long_context/v1.0-simplified_simplified-nq-train.jsonl"
    filtered_data_path = "/data/yjoonjang/datasets/long_context_dev/v1.0-simlified-simplified-nq-train-len=16k.json" # document_text 길이가 15.9K 이상 16K 이하인 문서들만 + long_answer 답이 <P> 태그로 시작하는 것들만
    total_rel_data_path = "/data/yjoonjang/datasets/long_context_dev/total_rel.json"
    total_rel_tailed_data_path = "/data/yjoonjang/datasets/long_context_dev/total_rel_tailed.json"
    document_texts_path = "/data/yjoonjang/datasets/long_context_dev/16k_rel_document_texts_rs=42.json"
    rel_data_16k_path = "/data/yjoonjang/datasets/long_context/16k_rel.json"
    rel_data_8k_path = "/data/yjoonjang/datasets/long_context/8k_rel.json"
    rel_data_4k_path = "/data/yjoonjang/datasets/long_context/4k_rel.json"
    unrel_data_16k_path = "/data/yjoonjang/datasets/long_context/16k_unrel.json"
    unrel_data_8k_path = "/data/yjoonjang/datasets/long_context/8k_unrel.json"
    unrel_data_4k_path = "/data/yjoonjang/datasets/long_context/4k_unrel.json"
    mixed_data_16k_path = "/data/yjoonjang/datasets/long_context/16k_mixed.json"
    mixed_data_8k_path = "/data/yjoonjang/datasets/long_context/8k_mixed.json"
    mixed_data_4k_path = "/data/yjoonjang/datasets/long_context/4k_mixed.json"


    # save_mixed(rel_data_8k_path, unrel_data_8k_path, mixed_data_8k_path, random_seed=42)
    # save_mixed(rel_data_4k_path, unrel_data_4k_path, mixed_data_4k_path, random_seed=42)
    # save_8k_4k(rel_data_16k_path, unrel_data_16k_path, rel_data_8k_path, unrel_data_8k_path, rel_data_4k_path, unrel_data_4k_path)











