# 来自 Llama 代码并进行了轻微修改
# 版权所有 (c) Meta Platforms, Inc. 和附属机构
# 本软件可根据 Llama 2 社区许可协议的条款使用和分发

import os
import struct
import argparse
from typing import List

from sentencepiece import SentencePieceProcessor

# 默认的 Llama SentencePiece 分词器模型文件名
TOKENIZER_MODEL = "tokenizer.model"

class Tokenizer:
    """
    Tokenizer 类：基于 SentencePiece 模型进行分词和解码操作，并支持导出分词器的二进制文件。
    """
    def __init__(self, tokenizer_model=None):
        """
        初始化分词器实例。

        :param tokenizer_model: 可选的分词器模型文件路径。如果未提供，则使用默认的 TOKENIZER_MODEL。
        """
        # 确定分词器模型路径
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path  # 检查模型文件是否存在

        # 加载 SentencePiece 模型
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # 获取模型中的特殊 token ID 和词汇表大小
        self.n_words: int = self.sp_model.vocab_size()  # 词汇表大小
        self.bos_id: int = self.sp_model.bos_id()       # BOS (句子开始) 的 ID
        self.eos_id: int = self.sp_model.eos_id()       # EOS (句子结束) 的 ID
        self.pad_id: int = self.sp_model.pad_id()       # PAD (填充符) 的 ID

        # 确保词汇表大小与分词器的 piece 数量一致
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        将字符串编码为 token ID 列表。

        :param s: 输入字符串。
        :param bos: 是否在编码结果前添加 BOS token。
        :param eos: 是否在编码结果后添加 EOS token。
        :return: 包含 token ID 的列表。
        """
        assert type(s) is str  # 确保输入是字符串
        t = self.sp_model.encode(s)  # 使用 SentencePiece 分词器进行分词
        if bos:
            t = [self.bos_id] + t  # 在列表开头添加 BOS token
        if eos:
            t = t + [self.eos_id]  # 在列表末尾添加 EOS token
        return t

    def decode(self, t: List[int]) -> str:
        """
        将 token ID 列表解码为字符串。

        :param t: 包含 token ID 的列表。
        :return: 解码后的字符串。
        """
        return self.sp_model.decode(t)  # 使用 SentencePiece 解码器将 token ID 转换为字符串

    def export(self):
        """
        导出分词器为二进制文件 `tokenizer.bin`。

        该文件包含分词器的 token 信息（包括其字节表示和得分），并记录了最大 token 长度。
        """

        # 存储所有 token（后处理后的）及其得分
        tokens, scores = [], []
        for i in range(self.n_words):
            # 获取 token 和其对应的得分
            t = self.sp_model.id_to_piece(i)  # 根据 ID 获取 token 字符串
            s = self.sp_model.get_score(i)    # 获取 token 的得分

            # 处理特殊 token：BOS 和 EOS
            if i == self.bos_id:
                t = '\n<s>\n'  # 将 BOS token 表示为特殊字符串
            elif i == self.eos_id:
                t = '\n</s>\n'  # 将 EOS token 表示为特殊字符串

            # 替换 SentencePiece 使用的空格符号（▁）为普通空格
            t = t.replace('▁', ' ')
            # 将 token 转换为 UTF-8 字节表示
            b = t.encode('utf-8')

            tokens.append(b)   # 添加到 tokens 列表
            scores.append(s)   # 添加到 scores 列表

        # 计算最大 token 长度
        max_token_length = max(len(t) for t in tokens)

        # 将数据写入二进制文件
        tokenizer_bin = self.model_path.replace('.model', '.bin')  # 生成二进制文件名
        with open(tokenizer_bin, 'wb') as f:
            # 写入最大 token 长度
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                # 写入每个 token 的得分和长度
                f.write(struct.pack("fI", score, len(bytes)))
                # 写入 token 的字节数据
                f.write(bytes)

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer-model", type=str, help="可选的自定义分词器路径")
    args = parser.parse_args()

    # 初始化分词器并导出二进制文件
    t = Tokenizer(args.tokenizer_model)
    t.export()