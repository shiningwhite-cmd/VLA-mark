import os
import argparse

import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor, AutoModel, AutoModelForCausalLM, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, LogitsProcessor, LogitsProcessorList
from transformers import LongformerModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers.generation.configuration_utils import GenerationConfig
import json
from tqdm import tqdm
import time

from vla import VLALogitsProcessor, VLA

import warnings
import logging

class WatermarkUtils:
    def __init__(self, params):
        self.config = params
        self._janus_attention_mask = None

    def init_model(self):
        model_map = {
            "llava": "llava-1.5-7b-hf",
            "qwen": "Qwen2-VL-7B-Instruct",
        }

        if self.config.model_name not in model_map:
            raise ValueError(f"Invalid model name: {self.config.model_name}")
            
        model_path = self.config.model_path

        if self.config.model_name == "llava":
            model = LlavaForConditionalGeneration.from_pretrained(model_path).to("cuda")
            processor = AutoProcessor.from_pretrained(model_path, patch_size=14)
        elif self.config.model_name == "qwen":
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, device_map="cuda")
            processor = AutoProcessor.from_pretrained(model_path, fast_image_processor_class=Qwen2VLImageProcessor)
        else:
            raise ValueError(f"Invalid model name: {self.config.model_name}")

        return model, processor

    def init_logits_processor(self, model, processor):
        special_tokens = [processor.tokenizer.eos_token_id, processor.tokenizer.bos_token_id] if processor.tokenizer.pad_token_id is None else [processor.tokenizer.eos_token_id, processor.tokenizer.bos_token_id, processor.tokenizer.pad_token_id]

        # 从模型获取词嵌入矩阵，并确保在 CUDA 上
        embedding_matrix = model.get_input_embeddings().weight
        if embedding_matrix.device.type != 'cuda':
            embedding_matrix = embedding_matrix.to('cuda')

        kwargs = {
            "embedding_matrix": embedding_matrix,
            "vocab_size": processor.tokenizer.vocab_size,
            "special_tokens": special_tokens,
            "similarity_scheme": self.config.similarity_scheme,
            "split_x": getattr(self.config, 'split_x', 2),
        }

        return VLALogitsProcessor(**kwargs)


    def init_detector(self, model, processor):
        embedding_matrix = model.get_input_embeddings().weight

        kwargs = {
            "processor": processor,
            "embedding_matrix": embedding_matrix,
            "is_hard": False,
            "similarity_scheme": self.config.similarity_scheme,
        }

        kwargs.update({
            "model": model,
            "model_name": self.config.model_name
        })

        return VLA(**kwargs)

    def _llava_responses(self, model, processor, inputs, logits_processor=None) -> str:
        generation_config = {
            "max_new_tokens": self.config.max_tokens,
            "min_new_tokens": self.config.min_tokens,
            "temperature": 0,
            "do_sample": False,
            "num_beams": 1,
        }
        
        if logits_processor is not None:
            generation_config["logits_processor"] = LogitsProcessorList([logits_processor])

        outputs = model.generate(**inputs, **generation_config)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        if "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:")[1].strip()

        return generated_text

    def _qwen_responses(self, model, processor, inputs, logits_processor=None) -> str:
        inputs = inputs.to('cuda')
        generation_config = {
            "max_new_tokens": self.config.max_tokens,
            "min_new_tokens": self.config.min_tokens,
            "temperature": 0,
            "do_sample": False,
            "num_beams": 1,
        }
        
        if logits_processor is not None:
            generation_config["logits_processor"] = LogitsProcessorList([logits_processor])

        outputs = model.generate(**inputs, **generation_config)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        if "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:")[1].strip()
        if "\nassistant\n" in generated_text:
            generated_text = generated_text.split("\nassistant\n")[1].strip()

        return generated_text

    def _image_feature(self, model, processor, inputs, image):
        if self.config.model_name == "llava":
            image_outputs = model.vision_tower(inputs['pixel_values'], output_hidden_states=True)
            image_feature = image_outputs.hidden_states[-2]
            image_feature = model.multi_modal_projector(image_feature)
            return image_feature
        elif self.config.model_name == "qwen":
            pixel_values = inputs['pixel_values'].to(model.dtype)
            image_grid_thw = inputs['image_grid_thw']
            visual_model = model.visual.to("cuda")
            pixel_values = pixel_values.to("cuda")
            image_embeds = visual_model(pixel_values, grid_thw=image_grid_thw)
            return image_embeds
        else:
            raise ValueError(f"Invalid model name: {self.config.model_name}")

    def get_feature(self, model, processor, inputs, image):
        return self._image_feature(model, processor, inputs, image)

    def get_cls_feature(self, model, processor, inputs, image):
        if self.config.model_name == "llava":
            image_outputs = model.vision_tower(inputs['pixel_values'], output_hidden_states=True)
            image_feature = image_outputs.hidden_states[-2]
            image_feature = model.multi_modal_projector(image_feature)
            image_feature = image_feature.squeeze(0)
            cls_feature = image_feature[0, :]
            return cls_feature
        elif self.config.model_name == "qwen":
            pixel_values = inputs['pixel_values'].to(model.dtype)
            image_grid_thw = inputs['image_grid_thw']
            visual_model = model.visual.to("cuda")
            pixel_values = pixel_values.to("cuda")
            image_embeds = visual_model(pixel_values, grid_thw=image_grid_thw)
            logging.info(f"image_embeds shape: {image_embeds.shape}")
            return image_embeds[0, 0, :]
        else:
            raise ValueError(f"Invalid model name: {self.config.model_name}")

    def get_inputs(self, model, processor, image, question):
        if self.config.model_name == "qwen":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{question}"},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt], images=[image], padding=True, return_tensors="pt"
            ).to("cuda")
        else:
            prompt = f"USER: <image>\n{question} ASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        return inputs

    def generate_responses(self, model, processor, inputs, logits_processor=None) -> str:
        response_map = {
            "qwen": self._qwen_responses,
            "llava": self._llava_responses,
        }
        
        response_func = response_map.get(self.config.model_name, self._llava_responses)
        return response_func(model, processor, inputs, logits_processor)
