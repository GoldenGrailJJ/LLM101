import os
from modelscope import snapshot_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from awq import AutoAWQForCausalLM

# 设置环境变量
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 路径和配置
BASE_DIR = "/root/autodl-tmp"
MODEL_NAME = "qwen/Qwen2-7B-Instruct"
QUANT_SUBDIR = "awq"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
QUANT_PATH = os.path.join(BASE_DIR, QUANT_SUBDIR)
QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

def model_download(model_name, cache_dir=BASE_DIR, revision="master"):
    """
    下载模型到指定目录
    """
    model_dir = snapshot_download(model_name, cache_dir=cache_dir, revision=revision)
    print(f"Model downloaded to: {model_dir}")
    return model_dir

def get_completion(prompts, model, tokenizer=None, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    """
    基于 vLLM 的文本生成函数
    """
    stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len, trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def quantize_and_save_model():
    """
    加载模型并量化后保存
    """
    # 加载模型和分词器
    model = AutoAWQForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 准备校准数据
    dataset = [
        [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are an expert in travel planning and programming."},
            {"role": "user", "content": "Can you recommend a scenic spot for a weekend trip near Beijing?"},
            {"role": "assistant", "content": "Certainly! The Great Wall at Mutianyu would be a perfect choice for a weekend getaway. It's less than two hours' drive from Beijing and offers stunning views and a less crowded experience compared to Badaling."}
        ],
        [
            {"role": "user", "content": "How about a good place for a family vacation in China?"},
            {"role": "assistant", "content": "For a family vacation, I would suggest Shanghai Disneyland. It offers a magical experience with various attractions and parades suitable for all ages. Plus, it's a great way to combine fun with a visit to one of China's most vibrant cities."}
        ],
        # 更多校准数据...
    ]
    data = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False).strip() for msg in dataset]

    # 模型量化
    model.quantize(
        tokenizer,
        quant_config=QUANT_CONFIG,
        calib_data=data,
        max_calib_seq_len=256
    )
    
    # 保存量化后的模型和分词器
    model.save_quantized(QUANT_PATH, safetensors=True, shard_size="4GB")
    tokenizer.save_pretrained(QUANT_PATH)

    print(f"Model is quantized and saved at '{QUANT_PATH}'")

if __name__ == "__main__":
    # 下载模型（如果需要）
    local_model_path = model_download(MODEL_NAME)

    # 基于 vLLM 的推理
    prompts = ["你好，帮我介绍一下什么是大语言模型。", "可以给我将一个有趣的童话故事吗？"]
    tokenizer = None  # 如果需要，可以加载分词器：AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    outputs = get_completion(prompts, MODEL_PATH, tokenizer=tokenizer, max_tokens=512, temperature=1, top_p=1, max_model_len=2048)
    
    # 打印生成结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    # 模型量化
    quantize_and_save_model()