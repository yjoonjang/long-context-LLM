from datasets import load_dataset
from tqdm import tqdm
from html.parser import HTMLParser
import json


class TextFromPParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.data = []

    def handle_starttag(self, tag, attrs):
        if tag == 'p':
            self.recording = True

    def handle_endtag(self, tag):
        if tag == 'p':
            self.recording = False

    def handle_data(self, data):
        if self.recording:
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


if __name__ == "__main__":
    # dataset = load_dataset("natural_questions")
    """
    NQ data
    - train, valid
    - keys: ['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])
    - if exists (long_answer & short_answers) & "yes_no_answer": "NONE"  --> data
    - Title: document_text 기준 맨 앞에 Wikipedia 전까지 
    - <P> </P> 안에 있는 것들이 document_text 인거임
    - 16K or 15.9K 중 100개 
    - Table 빼기
    """
    source_file = "/data/koo/datasets/long_context/v1.0-simplified_simplified-nq-train.jsonl"
    sampled_file = "/data/yjoonjang/datasets/long_context/v1.0-simlified-simplified-nq-train-sampled100.jsonl"

    # JSON Lines 파일에서 처음 100개의 데이터 읽어오기
    with open(source_file, "r") as source:
        count = 0
        while True:
            line = source.readline()
            if not line:
                break
            dataset = json.loads(line.strip())

            # check if 15.9K < len(document_text) < 16.1K & exists (long_answer & short_answers) & "yes_no_answer": "NONE"
            document_text = dataset["document_text"]
            annotations = dataset["annotations"]
            if ((15900 < len(document_text)) and (len(document_text) < 16100)) and (annotations[0]["yes_no_answer"] != "None") and (annotations[0]["long_answer"]) and (len(annotations[0]["short_answers"]) >= 1):
                title = extract_title(document_text)
                print(title)
                document_text_tokens = document_text.split()
                question = dataset["question_text"]
                extracted_document_text_list = extract_text_from_p(document_text)
                print(extracted_document_text_list)
                print(f"total length of document_text: {len(document_text)}")

                long_answer_info = dataset["annotations"][0]["long_answer"]
                short_answers_info = dataset["annotations"][0]["short_answers"]
                print(f"question: {question}")
                print(f"long_answer: {document_text_tokens[long_answer_info['start_token']:long_answer_info['end_token']]}")
                print("short_answers: ")
                for short_answer_info in short_answers_info:
                    print(document_text_tokens[short_answer_info['start_token']:short_answer_info['end_token']])
                break