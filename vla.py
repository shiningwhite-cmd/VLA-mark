import torch
from transformers import LogitsProcessor
from math import sqrt, exp
from markllm.watermark.base import BaseWatermark
import numpy as np
import time
import math

import PIL


class SimilarityScheme():
    def __init__(self, similarity_scheme: str) -> None:
        self.similarity_scheme = similarity_scheme
        self.similarity_scheme_map = {
            "cosine": self._cosine_similarity,
            "dot_product": self._dot_product_similarity,
        }
        
    def _cosine_similarity(self, m1: torch.FloatTensor, m2: torch.FloatTensor) -> torch.FloatTensor:
        """计算矩阵中每一行与其他行之间的余弦相似度, 并进行归一化"""
        # 保证输入为二维 (N, D) 与 (M, D)
        if m1.dim() == 1:
            m1 = m1.unsqueeze(0)
        if m2.dim() == 1:
            m2 = m2.unsqueeze(0)
        # 归一化计算，避免除零（加入极小值）
        m1_norm = torch.norm(m1, dim=1, keepdim=True).clamp_min(1e-12)
        m2_norm = torch.norm(m2, dim=1, keepdim=True).clamp_min(1e-12)
        dot_product = torch.matmul(m1, m2.T)
        norm_product = m1_norm * m2_norm.T
        return dot_product / norm_product

    def _dot_product_similarity(self, m1: torch.FloatTensor, m2: torch.FloatTensor) -> torch.FloatTensor:
        """计算矩阵中每一行与其他行之间的点积相似度"""
        # 保证输入为二维 (N, D) 与 (M, D)
        if m1.dim() == 1:
            m1 = m1.unsqueeze(0)
        if m2.dim() == 1:
            m2 = m2.unsqueeze(0)
        return torch.matmul(m1, m2.T)

    def similarity(self, m1: torch.FloatTensor, m2: torch.FloatTensor) -> torch.FloatTensor:
        """计算矩阵中每一行与其他行之间的相似度, 返回相似度矩阵
        - 输入张量可能是一维（单向量），此处统一扩展为二维
        - 自动对齐设备与数据类型，避免 dtype/device 不匹配问题
        """
        assert m1 is not None, "m1 should not be None"
        assert m2 is not None, "m2 should not be None"
        # 设备与 dtype 对齐到 m2（通常为词嵌入矩阵的设备与类型）
        target_device = m2.device
        target_dtype = m2.dtype if torch.is_floating_point(m2) else torch.float32
        if m1.device != target_device:
            m1 = m1.to(target_device)
        if m2.device != target_device:
            m2 = m2.to(target_device)
        if m1.dtype != target_dtype:
            m1 = m1.to(dtype=target_dtype)
        if m2.dtype != target_dtype:
            m2 = m2.to(dtype=target_dtype)
        # 维度对齐在具体实现内部也会做一次兜底处理
        return self.similarity_scheme_map[self.similarity_scheme](m1, m2)



class VLALogitsProcessor(LogitsProcessor):  
    def __init__(self, vocab_size: int, embedding_matrix: torch.FloatTensor, similarity_scheme: str = "cosine", input_embeddings: torch.LongTensor = None, special_tokens: list[int] = None, split_x: int = 2, *args, **kwargs) -> None:
        self.special_tokens = special_tokens
        self.embedding_matrix = embedding_matrix

        self.hash_key = 15485863
        self.vocab_size = vocab_size
        self.gamma = round(1 / split_x, 2)
        self.prefix_length = 1
        self.delta = 2.0

        self.similarity_scheme = SimilarityScheme(similarity_scheme)

        self.rng = torch.Generator(device='cuda')
        self.rng.manual_seed(self.hash_key)
        self.prf = torch.randperm(self.vocab_size, device='cuda', generator=self.rng)

        self.sim_rng = torch.Generator(device='cpu')  # 改为CPU设备

        self.msg = None
        self.list_ids = None
        self.similarity_list = None
        self.M_ids = None
        self.cls = None
        self.entropy_max = None
        self.entropy_current = None
        
        if input_embeddings is not None:
            self.refresh_msg(input_embeddings)

    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token time. Used in each token position calculation."""
        time_result = 1
        # 遍历输入序列的前 prefix_length 个 token ID，将它们相乘并累乘到 time_result 中
        for i in range(0, self.prefix_length):
            time_result *= input_ids[-1 - i].item()
        # !!int(time_result % self.vocab_size) 强制类型转换，不确定是否正确
        return int(self.prf[int(time_result % self.vocab_size)])

    def set_entropy_max(self, entropy_max: float) -> None:
        """设置最大熵值"""
        self.entropy_max = entropy_max

    def set_entropy_current(self, entropy_current: float) -> None:
        """设置当前熵值"""
        self.entropy_current = entropy_current
    
    def refresh_msg(self, input_embeddings: torch.LongTensor) -> None:
        """每一次进行新图片的MLLM生成文本的水印添加之前, 根据图片的embedding, 刷新msg和list_ids"""
        msg = input_embeddings.squeeze(0)
        self.msg = msg
        self.refresh_similarity_list()

    def refresh_cls(self, cls_feature: torch.LongTensor) -> None:
        """每一次进行新图片的MLLM生成文本的水印添加之前, 根据图片的embedding, 刷新cls"""
        self.cls = cls_feature
        

    def refresh_similarity_list(self) -> None:
        """刷新相似度列表"""
        self._calculate_similarity_list()

    def get_msg(self) -> torch.LongTensor:
        """获取msg"""
        return self.msg

    def _calculate_similarity_list(self) -> float:
        """计算相似度列表"""
        msg = self.msg
        similarity_matrix = self.similarity_scheme.similarity(msg, self.embedding_matrix).to("cuda")

        similarity_list = self._calculate_patch_wise_relevance(similarity_matrix) + self._calculate_global_relevance(similarity_matrix) + self._calculate_locality_sensitivity_relevance(similarity_matrix)
        similarity_list = self._normalize(similarity_list)

        self.similarity_list = similarity_list

    def _calculate_patch_wise_relevance(self, similarity_matrix: torch.LongTensor) -> torch.LongTensor:
        """计算patch-wise relevance"""
        patch_wise_relevance, _ = similarity_matrix.max(dim=0)
        patch_wise_relevance = torch.softmax(patch_wise_relevance, dim=0)
        return patch_wise_relevance

    def _calculate_global_relevance(self, similarity_matrix: torch.LongTensor) -> torch.LongTensor:
        """"""
        cls_similarity_list = self.similarity_scheme.similarity(self.cls, self.embedding_matrix)
        cls_similarity_list = cls_similarity_list.squeeze(0)
        cls_similarity_list = torch.softmax(cls_similarity_list, dim=0)
        return cls_similarity_list

    def _calculate_locality_sensitivity_relevance(self, similarity_matrix: torch.LongTensor) -> torch.LongTensor:
        """计算局部敏感性relevance"""
        max_similarity_list, _ = torch.max(similarity_matrix, dim=0)
        min_similarity_list, _ = torch.min(similarity_matrix, dim=0)
        locality_sensitivity_relevance = max_similarity_list - min_similarity_list
        locality_sensitivity_relevance = torch.softmax(locality_sensitivity_relevance, dim=0)
        return locality_sensitivity_relevance
    
    def _normalize(self, similarity_list: torch.LongTensor) -> torch.LongTensor:
        """归一化相似度列表"""
        min_value = similarity_list.min()
        max_value = similarity_list.max()
        denom = (max_value - min_value)
        denom = denom if denom != 0 else torch.tensor(1.0, device=similarity_list.device, dtype=similarity_list.dtype)
        normalized_tensor = (similarity_list - min_value) / denom
        return normalized_tensor

    def _similarity_normal_grouping(self, similarity_list: torch.LongTensor, position: int):
        """随机扰动项lambda"""
        entropy_max = self.entropy_max
        entropy_current = self.entropy_current
        similarity_list = self.similarity_list
        current_gamma = 0.025 - 0.02 * (self.entropy_current / self.entropy_max)

        # 计算高相似度词表
        sorted_indices = torch.argsort(similarity_list, descending=True)
        list_size = int(self.vocab_size * current_gamma)
        M_ids = sorted_indices[:list_size].tolist()

        # 生成随机排列，排除M_ids
        random_permutation_full = torch.randperm(self.vocab_size, generator=self.sim_rng).tolist()
        M_ids_set = set(M_ids)
        random_permutation = [i for i in random_permutation_full if i not in M_ids_set]
        
        # 分割随机词表
        split_index = int(self.vocab_size * 0.5 - len(M_ids))
        G_ids = random_permutation[:split_index]
        R_ids = random_permutation[split_index:]
        
        # 返回两个列表
        list_ids = [[] for _ in range(2)]
        list_ids[0] = M_ids + G_ids  # 绿色词表
        list_ids[1] = R_ids          # 红色词表
        
        return list_ids

    def similarity_grouping(self, similarity_list: torch.LongTensor, position: int) -> list[list[int]]:
        """随机化相似度列表"""
        self.sim_rng.manual_seed(position * self.hash_key)
        return self._similarity_normal_grouping(similarity_list, position)

    def _similarity_mean(self, similarity_matrix: torch.LongTensor):
        """计算相似度列表的均值"""
        return torch.mean(similarity_matrix, dim=0)

    def _similarity_max(self, similarity_matrix: torch.LongTensor):
        """计算相似度列表的最大值"""
        similarity_list, _ = torch.max(similarity_matrix, dim=0)
        similarity_list = self._normalize(similarity_list)
        return similarity_list

    def get_list_ids(self, msg: torch.LongTensor, position: int) -> list[list[int]]:
        """根据msg和embedding_matrix计算相似度列表, 并根据相似度列表进行分割词汇表并获取绿色词表"""
        similarity_list = self.similarity_list
        if similarity_list.shape[0] > self.vocab_size:
            similarity_list = similarity_list[:self.vocab_size]

        list_ids = self.similarity_grouping(similarity_list, position)

        return list_ids

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, list_token_ids, position: int) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        # 使用第一个列表（绿色词表）
        if len(list_token_ids) > 0 and len(list_token_ids[0]) > 0:
            green_tokens_mask[:, list_token_ids[0]] = True
        return green_tokens_mask

        return green_tokens_mask

    def detection_random_only_list_ids(self, position: int) -> list[int]:
        self.sim_rng.manual_seed(position * self.hash_key)

        random_permutation_full = torch.randperm(self.vocab_size, generator=self.sim_rng).tolist()
        list_ids = [[] for _ in range(2)]
        list_ids[0] = random_permutation_full[:int(self.vocab_size * 0.5)]
        list_ids[1] = random_permutation_full[int(self.vocab_size * 0.5):]
        return list_ids

    def detection_calc_greenlist_mask(self, scores: torch.FloatTensor, list_token_ids: list[int], input_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        position = self._f(input_ids.view(-1))
        return self._calc_greenlist_mask(scores=scores, list_token_ids=list_token_ids, position=position)

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens and output the biased scores."""
        inverted_mask = ~greenlist_mask.to("cuda")
        scores.add_(inverted_mask * (-greenlist_bias))
            
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.prefix_length:
            return scores

        position = self._f(input_ids[0])

        squeeze_scores = scores.squeeze().to(scores.device)
        squeeze_scores = torch.nan_to_num(
            squeeze_scores,
            nan=-10.0,
            posinf=None,
            neginf=-10.0
        ).to(dtype=scores.dtype)
        
        self.entropy_max = math.log(self.vocab_size)
        self.entropy_current = torch.distributions.Categorical(logits=squeeze_scores).entropy().item()

        list_token_ids = self.get_list_ids(self.msg, position)

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, list_token_ids=list_token_ids, position=position)
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)

        return scores

class VLA(BaseWatermark):
    logitsprocessor: VLALogitsProcessor = None
    tokenizer = None
    is_random_only = False
    is_random_only_annealing = False
    def __init__(self, processor = None, embedding_matrix: torch.FloatTensor = None, similarity_scheme: str = "cosine", input_embeddings: torch.LongTensor = None, model = None, tokenizer = None, transformers_config = None, split_x: int = 4, model_name: str = None, *args, **kwargs) -> None:
        if processor is not None:
            self.processor = processor
            special_tokens = [processor.tokenizer.eos_token_id, processor.tokenizer.bos_token_id] if processor.tokenizer.pad_token_id is None else [processor.tokenizer.eos_token_id, processor.tokenizer.bos_token_id, processor.tokenizer.pad_token_id]
            self.tokenizer = processor.tokenizer
            self.model = model

            self.logitsprocessor = VLALogitsProcessor(
                vocab_size=processor.tokenizer.vocab_size, 
                embedding_matrix=embedding_matrix, 
                similarity_scheme=similarity_scheme, 
                special_tokens=special_tokens, 
                split_x=split_x,
                )
        elif tokenizer is not None:
            self.tokenizer = tokenizer
            special_tokens = [tokenizer.eos_token_id, tokenizer.bos_token_id] if tokenizer.pad_token_id is None else [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id]
            self.model = model

            self.logitsprocessor = VLALogitsProcessor(
                vocab_size=tokenizer.vocab_size, 
                embedding_matrix=embedding_matrix,
                similarity_scheme=similarity_scheme,
                special_tokens=special_tokens, 
                split_x=split_x,
                )

        if input_embeddings is not None:
            self.refresh_logits_processor(input_embeddings=input_embeddings)

        if transformers_config is not None:
            self.generation_model = transformers_config.model
            self.generation_tokenizer = transformers_config.tokenizer
            self.gen_kwargs = transformers_config.gen_kwargs

        self.model_name = model_name

    def set_random_only_process(self) -> None:
        """设置随机扰动项lambda"""
        self.is_random_only = True

    def refresh_logits_processor(self, input_embeddings: torch.LongTensor, cls_feature: torch.LongTensor) -> None:
        """刷新msg"""
        self.logitsprocessor.refresh_cls(cls_feature)
        self.logitsprocessor.refresh_msg(input_embeddings)

    def compute_z_score(self, observed_count: int, T: int) -> float:
        """Compute z-score for the given observed count and total tokens."""
        expected_count = 0.5  # 绿色词表应该占词汇表的50%
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """生成水印文本"""
        inputs = torch.tensor(self.e_processor.encode(prompt))
        outputs = self.e_model(inputs.unsqueeze(0).to("cuda"))
        input_embeddings = outputs[0]

        self.refresh_logits_processor(input_embeddings=input_embeddings)

        generate_with_watermark = partial(
            self.model.generate,
            logits_processor=LogitsProcessorList([self.logitsprocessor]), 
            **self.gen_kwargs
        )

        encoded_prompt = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.model.device)
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        watermarked_text = self.tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]

        return watermarked_text


    def detect_watermark(self, text: str, image: PIL.Image.Image = None, return_dict: bool = False):
        """Detect watermark in the text."""

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encoded_text = inputs["input_ids"][0].to('cuda')
        num_tokens_scored = len(encoded_text) - self.logitsprocessor.prefix_length
        green_token_count = 0
        # 循环测试encodder text中每一个token是否在绿色词表中
        for idx in range(self.logitsprocessor.prefix_length+1, len(encoded_text)):
            # 计算logit向量
            with torch.no_grad():
                text_current = self.tokenizer.decode(encoded_text[:idx], skip_special_tokens=True)
                inputs_current = self.tokenizer(text_current, return_tensors="pt").to(self.model.device)
                outputs = self.model(**inputs_current)
                scores = (outputs.logits.to(self.model.device)).squeeze(0)[-1]
                scores = scores.unsqueeze(0)

            position = self.logitsprocessor._f(encoded_text[:idx])
            msg = self.logitsprocessor.get_msg()
            if self.is_random_only:
                list_token_ids = self.logitsprocessor.detection_random_only_list_ids(position)
            else:
                # scores 预处理
                squeeze_scores = scores.squeeze().to(scores.device)
                if squeeze_scores.min() == float('-inf') or squeeze_scores.min() == float('nan'):
                    squeeze_scores = torch.nan_to_num(squeeze_scores, neginf=-10)

                # 当前 scores 概率分布对数
                log_current_score = torch.nn.functional.log_softmax(squeeze_scores, dim=0)
                # 当前 scores 概率分布
                current_score = torch.exp(log_current_score)
                # 当前熵值
                self.logitsprocessor.set_entropy_current(-torch.sum(current_score * log_current_score).item())
                # 最大熵分数分布
                ave_score = torch.ones(self.tokenizer.vocab_size).to(scores.device)
                # 最大熵概率分布对数
                log_ave_score = torch.nn.functional.log_softmax(ave_score, dim=0)
                # 最大熵概率分布
                ave_score = torch.exp(log_ave_score)
                # 最大熵熵值
                self.logitsprocessor.set_entropy_max(-torch.sum(ave_score * log_ave_score).item())
                list_token_ids = self.logitsprocessor.get_list_ids(msg, position)
            green_tokens_mask = self.logitsprocessor.detection_calc_greenlist_mask(
                    scores=scores, 
                    list_token_ids=list_token_ids,
                    input_ids=encoded_text[:idx]
                )
            if bool(green_tokens_mask[0][int(encoded_text[idx])]) is True:
                green_token_count += 1
            
        z_score = self.compute_z_score(green_token_count, num_tokens_scored)
        # 3.Determine if the z_score indicates a watermark
        is_watermarked = z_score > 4.0

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "z_score": z_score}
        else:
            return (is_watermarked, z_score)
