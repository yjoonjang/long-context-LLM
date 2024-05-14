from datasets import load_dataset
from tqdm import tqdm
from html.parser import HTMLParser
import json
import jsonlines
import random
import matplotlib.pyplot as plt
from get_similarity_score import get_embedding, store_vector


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


if __name__ == "__main__":
    source_file = "/data/koo/datasets/long_context/v1.0-simplified_simplified-nq-train.jsonl"
    filtered_data_path = "/data/yjoonjang/datasets/long_context_dev/v1.0-simlified-simplified-nq-train-len=16k.json" # document_text 길이가 15.9K 이상 16K 이하인 문서들만 + long_answer 답이 <P> 태그로 시작하는 것들만
    total_rel_data_path = "/data/yjoonjang/datasets/long_context/total_rel.json"
    total_rel_tailed_data_path = "/data/yjoonjang/datasets/long_context/total_rel_tailed.json"
    rel_data_path = "/data/yjoonjang/datasets/long_context/16k_rel.json"
    document_texts_path = "/data/yjoonjang/datasets/long_context_dev/16k_rel_document_texts_rs=42.json"
    unrel_data_path = "/data/yjoonjang/datasets/long_context/16k_unrel.json"

    with open(document_texts_path, 'r') as f:
        document_texts = json.load(f)
        vectors = []
        for i in range(len(document_texts)):
            values = get_embedding(document_texts[i])
            vector_id = f"{i}"
            metadata = {
                "rel_document_text_index": i
            }
            vector_dict = format_to_vector_dict(vector_id, values, metadata)
            vectors.append(vector_dict)
        store_vector(vectors)

    # with open(rel_data_path, 'r') as f:
    #     source_datasets = json.load(f)
    #     queries = []
    #     passages = []
    #     formatted_unrel_datasets = []
    #
    #     # task = 'Given a query, retrieve relevant passages that answer the query'
    #     for i, source_dataset in enumerate(source_datasets):
    #         question = source_dataset["question_text"]
    #         documents = source_dataset["document_text"]
    #
    #         # Each query must come with a one-sentence instruction that describes the task
    #         queries.append(get_detailed_instruct(task, question))
    #         passages.append(documents)
    #     lowest_score_indices = get_similarity_score_batched(queries, passages)
    #
    #     for i, source_dataset in enumerate(source_datasets):
    #         title = source_dataset["title"]
    #         document_text = source_dataset[lowest_score_indices[i]]["document_text"]
    #         question_text = source_dataset["question_text"]
    #         annotations = source_dataset["annotations"]
    #         document_url = source_dataset["document_url"]
    #         example_id = f"{i}_unrel"
    #
    #         formatted_unrel_data = {
    #             "title": title,
    #             "document_text": document_text,
    #             "question_text": question_text,
    #             "annotations": annotations,
    #             "document_url": document_url,
    #             "example_id": example_id
    #         }
    #
    #         formatted_unrel_datasets.append(formatted_unrel_data)
    #     save_json(unrel_data_path, formatted_unrel_datasets)







