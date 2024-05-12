from datasets import load_dataset
from tqdm import tqdm
from html.parser import HTMLParser
import json
import re


class TextFromPParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.data = []
        self.p_tag_open = False  # <p> 태그가 열려 있는지 확인하기 위한 플래그

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'p':
            self.recording = True
            self.p_tag_open = True  # <p> 태그가 시작됨

    def handle_endtag(self, tag):
        if tag.lower() == 'p':
            self.recording = False
            self.p_tag_open = False  # <p> 태그가 종료됨

    def handle_data(self, data):
        if self.recording and self.p_tag_open:
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


def get_excluded_indices(document_texts, long_answer):
    indices = list(range(1, len(document_texts) + 1))

    # long_answer에 해당하는 인덱스를 제외
    excluded_indices = [index for index in indices if document_texts[index - 1] != long_answer]

    return excluded_indices


if __name__ == "__main__":
    # dataset = load_dataset("natural_questions")
    """
    NQ data
    - train, valid
    - keys: ['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])
    - if exists (long_answer & short_answers) & "yes_no_answer": "NONE"  --> data
    - Title: document_text 기준 맨 앞에 Wikipedia 전까지 
    - <P> </P> 안에 있는 것들이 document_text 인거임
    - 16K (~15.9K) 중 100개 
    - Table 빼기
    """
    source_file = "/data/koo/datasets/long_context/v1.0-simplified_simplified-nq-train.jsonl"
    filtered_data = "datasets/v1.0-simlified-simplified-nq-train-len=16k.jsonl" # 길이가 15.9K 이상 16K 이하인 문서들 길이로 필터링

    with open(filtered_data, "r") as source:
        while True:
            line = source.readline()
            if not line:
                break

            dataset = json.loads(line.strip())
            document_text = dataset["document_text"]
            annotations = dataset["annotations"]

            # check if 15.9K < len(document_text) < 16.1K & exists (long_answer & short_answers) & "yes_no_answer": "NONE"
            # if ((15900 <= len(document_text)) and (len(document_text) <= 16000)) and (annotations[0]["yes_no_answer"] != "None") and (annotations[0]["long_answer"]) and (len(annotations[0]["short_answers"]) >= 1):
            document_text_tokens = document_text.split() # document_text를 토큰화

            title = extract_title(document_text)
            question_text = dataset["question_text"]
            extracted_document_texts = extract_text_from_p(document_text)
            banned_text_list = [" ", " . ", "<", "  "]

            # 필터링
            extracted_document_texts = [text for text in extracted_document_texts if text not in banned_text_list]

            long_answer_info = dataset["annotations"][0]["long_answer"]
            short_answers_info = dataset["annotations"][0]["short_answers"]

            short_answers = []
            long_answer = " ".join(document_text_tokens[long_answer_info['start_token']:long_answer_info['end_token']])
            long_answer = extract_text_from_p(long_answer)[0] # long_answer에서 P 태그 제거
            for short_answer_info in short_answers_info:
                short_answer = document_text_tokens[short_answer_info['start_token']:short_answer_info['end_token']]
                short_answers.append(" ".join(short_answer))

            indices_list = get_excluded_indices(extracted_document_texts, long_answer)

            print(f"total length of document_text: {len(document_text)}")
            print(f"title: {title}")
            print(f"document_text: {extracted_document_texts}")
            print(f"related_information: {indices_list}")
            print(f"question_text: {question_text}")
            print(f"long_answer: {long_answer}")
            print(f"short_answers: {short_answers}")
            break