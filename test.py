import requests
import json
import time
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict
import base64
import mimetypes
from tqdm import tqdm

from open_source_config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_REF_DIR,
    DEFAULT_RESULT_DIR,
    build_headers,
    get_api_config,
)

DEFAULT_BASEURL_ENV = "POLYREAL_API_BASE_URL"
DEFAULT_API_KEY_ENV = "POLYREAL_API_KEY"
INTERN_S1_BASEURL_ENV = "INTERN_S1_API_BASE_URL"
INTERN_S1_API_KEY_ENV = "INTERN_S1_API_KEY"

# 全局变量，将在主程序中根据模型名称动态设置
url = None
headers = None

image_dir = ""
lock = Lock()
output_file = None
error_file = None

def setup_api_config(model_name: str):
    """根据模型名称设置 API 配置"""
    global url, headers
    
    if model_name == "intern-s1":
        base_url, api_key = get_api_config(INTERN_S1_BASEURL_ENV, INTERN_S1_API_KEY_ENV)
    else:
        base_url, api_key = get_api_config(DEFAULT_BASEURL_ENV, DEFAULT_API_KEY_ENV)
    
    url = base_url + "/v1/chat/completions"
    headers = build_headers(api_key)

def to_data_url(path: str) -> str:
    """将本地图片文件转换为 Base64 编码的 Data URL。"""
    if not os.path.exists(path):
        print(f"警告: 找不到图片文件 {path}")
        return ""
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"  # 默认 MIME 类型
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def parse_llm_response(response_text: str) -> dict:
    """
    从LLM的完整响应中解析出 <think> 和 <answer> 标签的内容。
    """
    think_content = ""
    answer_content = ""

    try:
        # 提取 <think> 标签内容
        if "<think>" in response_text and "</think>" in response_text:
            think_start = response_text.find("<think>") + len("<think>")
            think_end = response_text.find("</think>")
            think_content = response_text[think_start:think_end].strip()
    except Exception as e:
        print(f"解析 <think> 标签时出错: {e}")

    try:
        if "<answer>" in response_text:
            answer_start = response_text.find("<answer>") + len("<answer>")            
            answer_end = response_text.find("</answer>", answer_start)
            if answer_end != -1:
                answer_content = response_text[answer_start:answer_end]
            else:
                answer_content = response_text[answer_start:]
            answer_content = answer_content.strip()
    except Exception as e:
        print(f"解析 <answer> 标签时出错: {e}")
        
    return {
        "think": think_content,
        "answer": answer_content
    }

def extract_json_from_llm_output(llm_output: str) -> dict:
    """
    从 LLM 输出中提取 JSON 对象，自动去除 markdown 包裹等问题。
    """
    import re
    text = llm_output.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def get_answer_threadsafe(idx, item, max_retries=3):
    """
    线程安全函数，用于获取单个条目的LLM回答，并支持重试。
    idx: 条目的唯一ID。
    item: 包含Question, Path等信息的字典。
    max_retries: 最大重试次数。
    """
    question_text = item.get("Question", "")
    gt_answer = item.get("Answer", "")
    image_path_suffix = item.get("Path", "")
    Keywords = item.get("Keywords", "")
    
    # 判断是否为 ranking 任务 (id 在 472-505 范围内)
    is_ranking_task = 472 <= idx <= 505
    
    # 根据任务类型选择不同的 system prompt
    if is_ranking_task:
        system_content = """You are a polymer science expert. Your task is to analyze the provided question and provide a reasoned, accurate answer.

        You MUST respond ONLY with a valid JSON object. Do not include any text, explanations, or markdown formatting (like ```json) before or after the JSON structure.

        Your JSON output must contain exactly two keys:
        1.  "llm_think": A string containing your detailed, step-by-step reasoning to arrive at the solution.
        2.  "llm_answer": This field MUST be a JSON list (array). The contents of the list depend on the question type:
            - For sorting : A list of strings (e.g., ["c", "b", "a"] ).

        Example for a sorting question:
        {
        "llm_think": "The question asks to sort... High proportion means... low proportion means... Therefore the order is C, then B, then A.",
        "llm_answer": ["c", "b", "a"]
        }"""
    else:
        system_content = """You are a polymer science expert. Your task is to provide a clear and accurate answer.
            Process:

            Internal Reasoning (inside the <think></think> tag): Lay out your step-by-step reasoning process here.

            Final Synthesized Answer (inside the <answer></answer> tag): After your reasoning, place the well-organized, clear, accurate, and concise answer within the <answer></answer> tag. This answer must be a standalone, concise, and professional explanation that directly addresses the user's question. Do not simply repeat the reasoning process. You should distill the key conclusions from your thinking process to form a polished response.

            Please ensure your final response includes both the complete <think> and <answer> sections."""
    
    # 动态构建 API 请求的 payload
    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": []
        }
    ]

    # 添加文本部分
    if question_text:
        messages[1]["content"].append({"type": "text", "text": question_text})

    # 如果有图片路径，则处理并添加图片部分
    if image_path_suffix:
        full_file_path = os.path.join(image_dir, image_path_suffix)

        if not os.path.exists(full_file_path):
            print(f"警告: 在路径 {full_file_path} 找不到文件，跳过该文件。")
        elif full_file_path.lower().endswith('.csv'):
            try:
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    csv_content = f.read()
                messages[1]["content"].append({
                    "type": "text",
                    "text": f"\n\n--- Attached CSV file content ({image_path_suffix}) ---\n{csv_content}"
                })
                print(f"已成功读取并添加CSV文件: {image_path_suffix}")
            except Exception as e:
                print(f"读取CSV文件 {full_file_path} 时出错: {e}")
        else:
            img_data_url = to_data_url(full_file_path)
            if img_data_url:  
                messages[1]["content"].append(
                    {"type": "image_url", "image_url": {"url": img_data_url, "detail": "auto"}}
                )
    
    payload = {
        "model": model_name,
        "messages": messages,
        "thinking_mode": True
    }
    # -初始化变量 ---
    llm_response = "" # 初始化原始响应变量
    llm_think = ""    # 初始化
    llm_answer = "" if not is_ranking_task else []   # 初始化：ranking 任务为列表，其他为字符串
    elapsed = 0       # 初始化

    reminder_added = False  # 标记是否已添加提示
    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=3000) # 使用 json=payload
            response.raise_for_status() 
            data = response.json()
            # elapsed = time.time() - start_time
            llm_response = data['choices'][0]['message']['content']
            
            if is_ranking_task:
                # Ranking 任务：使用 JSON 解析
                parsed_json = extract_json_from_llm_output(llm_response)
                think_content = parsed_json.get("llm_think")
                answer_content = parsed_json.get("llm_answer")
                if (think_content and 
                    answer_content is not None and 
                    isinstance(answer_content, list)):
                    llm_think = think_content
                    llm_answer = answer_content
                    elapsed = time.time() - start_time
                    break
                else:
                    raise ValueError(f"ID {idx} 内容校验失败: JSON中缺少 'llm_think' 或 'llm_answer' (list)。Got: {llm_response}")
            else:
                # 普通任务：使用 XML 标签解析
                parsed_data = parse_llm_response(llm_response)
                llm_think = parsed_data["think"]
                llm_answer = parsed_data["answer"]
                if llm_answer:
                    elapsed = time.time() - start_time
                    break 
                if reminder_added:
                    pass
                else:
                    # 在错误信息中加入 idx
                    raise ValueError(f"ID {idx} 内容校验失败: 响应中缺少 <think> 标签内容。{llm_response}")
            # break  
        except Exception as e:
            elapsed = time.time() - start_time
            llm_response_error = f"Error (attempt {attempt}/{max_retries}): {str(e)}"
            print(f"索引 {idx} 在第 {attempt} 次尝试时出错{e}")
            
            # 根据任务类型选择不同的错误处理
            if is_ranking_task:
                if (isinstance(e, (json.JSONDecodeError, ValueError))) and not reminder_added:
                    reminder_prompt = {
                        "type": "text",
                        "text": "\n\n--- \n[System Reminder]: Your previous response was not a valid JSON or missed required keys. You MUST respond ONLY with a valid JSON object containing 'llm_think' (string) and 'llm_answer' (list)."
                    }
                    messages[1]["content"].append(reminder_prompt)
                    reminder_added = True
                    print(f"索引 {idx}: 已为下一次重试动态添加 JSON 格式提醒。")
            else:
                if isinstance(e, ValueError) and not reminder_added:
                    # 这是一个追加到 user message content 列表中的新 text 块
                    reminder_prompt = {
                        "type": "text",
                        "text": "\n\n--- \n[System Reminder]: Your previous response was missing the required format. Please try again and ensure your final output strictly contains both a `<think>...</think>` section and an `<answer>...</answer>` section."
                    }
                    messages[1]["content"].append(reminder_prompt)
                    reminder_added = True  # 将标志位设为True，确保提示只添加一次
                    print(f"索引 {idx}: 已为下一次重试动态添加格式提醒。")
                    # print(messages)            
            if attempt < max_retries:
                time.sleep(10)
            else:
                print(f"索引 {idx} 在达到最大重试次数后仍然失败。")
                break 

    record = {
        "id": idx, # 使用原始的id
        "Question": question_text,
        "gt_answer": gt_answer,
        "llm_response": llm_response, # 存储完整的原始回答
        "llm_think": llm_think,     # 存储提取的思考过程
        "llm_answer": llm_answer,      # 存储提取的最终答案
        "Keywords": Keywords,
        "elapsed_time": round(elapsed, 2)
    }
    # is_success = bool(llm_think and llm_answer)
    is_success = bool(llm_answer)
    target_file = output_file if is_success else error_file

    with lock:
        with open(target_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    return idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM on PolyREAL dataset with concurrency")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model name to use")#gpt-4o gemini-2.5-flash gemini-2.5-pro grok-4  gpt-5 x-ai/grok-4 gpt-4o-mini meta-llama/llama-3.1-70b-instruct deepseek-r1 
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent threads")
    parser.add_argument("--input_file", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to the input JSON file")
    parser.add_argument("--image_dir", type=str, default=str(DEFAULT_REF_DIR), help="Directory containing image and CSV references")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_RESULT_DIR), help="Directory to save the output file")
    args = parser.parse_args()

    image_dir = args.image_dir
    model_name = args.model
    num_workers = args.workers
    input_file_path = args.input_file

    # 根据模型名称设置 API 配置
    setup_api_config(model_name)

    safe_model_name = model_name.replace("/", "_")
   
    args.output_dir = args.output_dir + "/" + safe_model_name

    os.makedirs(args.output_dir, exist_ok=True) 
    output_file = os.path.join(args.output_dir, f"results_{safe_model_name}.jsonl")
    error_file = os.path.join(args.output_dir, f"errors_{safe_model_name}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True) 
    print(f"模型: {model_name}")
    print(f"输入文件: {input_file_path}")
    print(f"输出将保存至: {output_file}")
    print(f"错误记录将保存至: {error_file}")
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"成功加载数据集，共 {len(dataset)} 条记录。")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误：输入文件 {input_file_path} 不是有效的 JSON 格式。")
        exit(1)

    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data and not data.get("llm_response", "").startswith("Error"):
                        processed_ids.add(data["id"])
                except json.JSONDecodeError:
                    print(f"警告：跳过格式错误的行: {line.strip()}")
        print(f"检测到已有 {len(processed_ids)} 条成功处理的记录。")
        print(processed_ids)
    
    tasks = [(item.get("id"), item) for item in dataset if item.get("id") not in processed_ids]
    
    if not tasks:
        print("所有任务已完成！程序退出。")
        exit(0)
    
    print(f"待处理任务数量: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(get_answer_threadsafe, idx, item) for idx, item in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"并发处理 ({model_name})"):
            try:
                future.result()
            except Exception as e:
                print(f"一个线程任务执行时发生严重错误: {e}")

    print(f"\n✅ 所有处理完成，结果保存在 {output_file}")
