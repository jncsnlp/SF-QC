from ollama import chat
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import httpx
import csv
import re
import random
import os
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Any, Sequence
import time

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_texts(texts, tokenizer, model):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(sentence_embeddings, p=2, dim=1)

def calculate_top_k_labels(input_text, labels, tokenizer, model, k):
    target_embedding = encode_texts([input_text], tokenizer, model)
    label_embeddings = encode_texts(labels, tokenizer, model)
    cosine_scores = torch.mm(target_embedding, label_embeddings.T).squeeze(0)
    top_k_indices = torch.topk(cosine_scores, k=k).indices
    return [labels[i] for i in top_k_indices]

def SFQC(input_text, labels, tokenizer, model, llm, k):
    # SF module
    dynamic_labels = calculate_top_k_labels(input_text, labels, tokenizer, model, k)

    # Without SF module
    # dynamic_labels = labels

    # Random shuffle
    # dynamic_labels = labels
    # random.seed(15)
    # random.shuffle(dynamic_labels)

    # Reverse
    # dynamic_labels = labels
    # dynamic_labels.reverse()

    if llm == "qwen":
        llm = "qwen2.5:7b"
    elif llm == "deepseek":
        llm = "deepseek-r1:7b"
    elif llm == "llama":
        llm = "llama3:8b"
    ChatResponse1 = chat(model=llm, messages=[
              {"role": "user", "content": f'''
                Sentence: {input_text}
                Supported Intents: {dynamic_labels}
                Task: Identify the intent of the sentence. If the intent matches one from the Supported Intents, choose that option. Otherwise, choose "unknown". Let's think step by step.

                <Response format>
                Please respond to me with the format of "Intent: XX" or "Intent: unknown"
              '''},],options={'temperature': 1})

    model_reply = ChatResponse1['message']['content']
    pattern = r"Intent:\s*(\w+)"
    match = re.search(pattern, model_reply)
    intent_label = match.group(1)
    print(f"Extracted Intent: {intent_label}")
    if intent_label == "unknown":
        return intent_label,intent_label
    text3 = f'''
                Question 1: Is the sentence's purpose closely related to the intent "{intent_label}"?
                Question 2: Is "{intent_label}" the most suitable intent available, compared to other options?
                Question 3: Is "{intent_label}" a better match for this sentence than "unknown"?

                Response Format:
                "Answer 1: XX, Answer 2: XX, Answer 3: XX"
                ''' 
    ChatResponse2 = chat(model=llm, messages=[
          {"role": "user", "content": f'''
                Sentence: {input_text}
                Supported Intents: {dynamic_labels}
                Task: Identify the intent of the sentence. If the intent matches one from the Supported Intents, choose that option. Otherwise, choose "unknown". Let's think step by step.

                <Response format>
                Please respond to me with the format of "Intent: XX" or "Intent: unknown"
          '''},
          {"role": "assistant", "content": f"{intent_label}"},
          {"role": "user", "content": f"{text3}"},
        ]
      ,options={'temperature': 1})
    model_reply1 = ChatResponse2['message']['content']

    text4 = f'''
                If the answers to the above questions are all "yes", directly return the intent. If any answer is "no", reconsider your response carefully. If, after reconsideration, you still cannot identify a suitable intent from the supported list, return "unknown".
                
                <Response format>
                "Intent: XX"'''
    ChatResponse3 = chat(model=llm, messages=[
          {"role": "user", "content": f'''
                Sentence: {input_text}
                Supported Intents: {dynamic_labels}
                Task: Identify the intent of the sentence. If the intent matches one from the Supported Intents, choose that option. Otherwise, choose "unknown". Let's think step by step.

                <Response format>
                Please respond to me with the format of "Intent: XX" or "Intent: unknown"
          '''},
          {"role": "assistant", "content": f"{intent_label}"},
          {"role": "user", "content": f"{text3}"},
          {"role": "assistant", "content": f"{model_reply1}"},
          {"role": "user", "content": f"{text4}"},
        ]
      ,options={'temperature': 1})
    model_reply2 = ChatResponse3['message']['content']
    pattern = r"Intent:\s*(\w+)"
    match = re.search(pattern, model_reply2)
    intent_label1 = match.group(1)
    return intent_label,intent_label1

def CoT(input_text, labels, llm):
    if llm == "qwen":
        llm = "qwen2.5:7b"
    elif llm == "deepseek":
        llm = "deepseek-r1:7b"
    elif llm == "llama":
        llm = "llama3:8b"    
    ChatResponse1 = chat(model=llm, messages=[
              {"role": "user", "content": f'''
                Sentence: {input_text}
                Supported Intents: {labels}
                Task: Identify the intent of the sentence. If the intent matches one from the Supported Intents, choose that option. Otherwise, choose "unknown". Let's think step by step.

                <Response format>
                Please respond to me with the format of "Intent: XX" or "Intent: unknown"
              '''},],options={'temperature': 1})

    model_reply = ChatResponse1['message']['content']
    pattern = r"Intent:\s*(\w+)"
    match = re.search(pattern, model_reply)
    intent_label = match.group(1)
    
    return intent_label

def run(
    task: str,
    selected_labels: Sequence[str],
    SF_ratio: float = 0.1,
    method: str = "SF-QC",
    llm: str = "qwen",
    seed: int = 15,
) -> dict[str, Any]:
    input_path = Path(f"./data/{task}/test.tsv")
    timestamp = time.strftime("%m%d%H%M")
    output_path = Path(f"./results/{llm}_{method}_{task}_{timestamp}_results.csv")
    error_path = Path(f"./results/{llm}_{method}_{task}_{timestamp}_error.txt")

    if not input_path.is_file():
        raise FileNotFoundError(f"{input_path} does not exist.")

    if not 0 < SF_ratio <= 1:
        raise ValueError("SF_ratio must be in the (0, 1] interval")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    model_name_or_path = f"./models/all-MiniLM-L6-v2"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
    )
    model.eval()

    m = min(
        len(selected_labels),
        max(1, int(len(selected_labels) * SF_ratio)),
    )

    total_count = 0
    success_count = 0
    failed_count = 0

    with (
        input_path.open("r", encoding="utf-8", newline="") as input_file,
        output_path.open("w", encoding="utf-8", newline="") as output_file,
        error_path.open("a", encoding="utf-8") as error_file,
    ):
        reader = csv.reader(input_file, delimiter="\t")
        if method == "SF-QC":
            writer = csv.writer(output_file)
            writer.writerow(["label", "SF_answer", "answer"])
        elif method == "CoT":
            writer = csv.writer(output_file)
            writer.writerow(["label", "answer"])

        next(reader, None)
        idx = 0
        for line_number, row in tqdm(
            enumerate(reader, start=2),
            desc="Running",
        ):
            if idx >= 1000:
                break
            idx += 1
            total_count += 1

            if len(row) < 2:
                failed_count += 1
                error_file.write(
                    f"Invalid row at line {line_number}: expected at least 2 columns, "
                    f"got {len(row)}. Row={row!r}\n"
                )
                continue

            text = row[0].strip()
            label = row[1].strip()

            if not text:
                failed_count += 1
                error_file.write(
                    f"Invalid row at line {line_number}: text is empty. "
                    f"Label={label!r}\n"
                )
                continue
            if method == "SF-QC":
                try:
                    SF_answer, SFQC_answer = SFQC(
                        text,
                        selected_labels,
                        tokenizer,
                        model,
                        llm=llm,
                        k=m,
                    )
                except Exception as exc:
                    failed_count += 1
                    error_file.write(
                        f"Error at line {line_number}. "
                        f"Input={text!r}. Actual label={label!r}. "
                        f"Error={type(exc).__name__}: {exc}\n"
                    )
                    error_file.flush()
                    continue
                if label not in selected_labels:
                    label = "unknown"
                print(f"SF: {SF_answer}, SFQC: {SFQC_answer}, Actual: {label}")
                writer.writerow([label, SF_answer, SFQC_answer])
                success_count += 1
            elif method == "CoT":
                try:
                    answer = CoT(
                        text,
                        selected_labels,
                        llm=llm,
                    )
                except Exception as exc:
                    failed_count += 1
                    error_file.write(
                        f"Error at line {line_number}. "
                        f"Input={text!r}. Actual label={label!r}. "
                        f"Error={type(exc).__name__}: {exc}\n"
                    )
                    error_file.flush()
                    continue
                if label not in selected_labels:
                    label = "unknown"
                print(f"CoT: {answer}, Actual: {label}")
                writer.writerow([label, answer])
                success_count += 1

    result = {
        "total": total_count,
        "success": success_count,
        "failed": failed_count,
        "label_count": len(selected_labels),
        "output_file": str(output_path),
        "error_log_file": str(error_path),
    }
    
    print(
        "Evaluation finished: "
        f"total={total_count}, success={success_count}, failed={failed_count}, "
    )
    print(f"Data saved to {output_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SF-QC OOD Intent Detection Evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="banking",
        choices=["banking", "clinc"],
        help="Dataset name. You can choose from banking, clinc.",
    )
    parser.add_argument(
        "--id_ratio",
        type=float,
        default=0.25,
        help="ID label ratio.",
    )
    parser.add_argument(
        "--SF_ratio",
        type=float,
        default=0.1,
        help="selected intents ratio.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["qwen", "deepseek", "llama"],
        default="qwen",
        help="LLM name. You can choose from qwen, deepseek, llama.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["SF-QC", "CoT"],
        default="SF-QC",
        help="Method. You can choose from SF-QC, CoT.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=15,
        help="Random seed.",
    )

    return parser.parse_args()

def get_selected_labels(task: str, id_ratio: float = 0.25):

    random.seed(15)
    data_by_label = {}
    data_path = f"./data/{task}/train.tsv" 
    with open(data_path, "r", encoding="utf-8") as file:
        next(file)  # 跳过标题行
        for line in file:
            data, label = line.strip().split("\t")
            if label not in data_by_label:
                data_by_label[label] = []
            data_by_label[label].append(data)

    # 随机选择25%的标签
    all_labels = list(data_by_label.keys())
    num_labels_to_select = max(1, int(len(all_labels) * id_ratio))
    selected_labels = random.sample(all_labels, num_labels_to_select)

    return selected_labels

def main() -> None:
    args = parse_args()

    selected_labels = get_selected_labels(args.task, args.id_ratio)

    error_log = f"{args.llm}_{args.task}_{args.method}_error.txt"

    run(
        task=args.task,
        selected_labels=selected_labels,
        SF_ratio=args.SF_ratio,
        method=args.method,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
