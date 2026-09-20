"""Local Qwen PE adapter; official profiles, streamed NF4 weights, one GPU job."""
from __future__ import annotations

import gc
import json
from pathlib import Path

from .vendor.qwen_pe import pe_core as core


def checkpoint_for(model_dir: Path, images):
    task = "edit" if images else "t2i"
    suffix = "I2I" if images else "T2I"
    return task, model_dir.parent / f"Qwen-Image-2.1-PE-{suffix}"


def available(model_dir: Path):
    result = {}
    for task, suffix in [("t2i", "T2I"), ("edit", "I2I")]:
        root = model_dir.parent / f"Qwen-Image-2.1-PE-{suffix}"
        try:
            result[task] = (json.loads((root / "download-status.json").read_text())["state"] == "completed"
                            and (root / "system_prompt.txt").is_file())
        except (OSError, ValueError, KeyError):
            result[task] = False
    return result


def enhance(model_dir, prompt, images, seed, update, cancelled):
    import torch
    from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig,
                              LogitsProcessor, LogitsProcessorList, StoppingCriteria,
                              StoppingCriteriaList)

    task, root = checkpoint_for(model_dir, images)
    if not available(model_dir)[task]:
        raise ValueError("The official prompt enhancer is still downloading")
    profile = core.get_profile(task)
    system = core.load_system_prompt(None, str(root))
    model = processor = inputs = output = None
    try:
        update(stage=f"Loading {'editing' if images else 'creation'} prompt enhancer")
        processor = AutoProcessor.from_pretrained(str(root), local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(root), local_files_only=True, low_cpu_mem_usage=True, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)).eval()
        if cancelled():
            return None
        references = []
        for source in images:
            image = source.convert("RGB")
            if image.width * image.height > profile.image_max_pixels:
                scale = (profile.image_max_pixels / (image.width * image.height)) ** .5
                image = image.resize((int(image.width * scale), int(image.height * scale)))
            references.append(image)
        messages = core.build_messages(system, prompt, references)
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=True).to(model.device)
        if "mm_token_type_ids" not in inputs and hasattr(processor, "create_mm_token_type_ids"):
            inputs["mm_token_type_ids"] = processor.create_mm_token_type_ids(inputs["input_ids"])
        prompt_len = inputs["input_ids"].shape[1]

        class PresencePenalty(LogitsProcessor):
            def __call__(self, input_ids, scores):
                for batch in range(input_ids.shape[0]):
                    generated = input_ids[batch, prompt_len:]
                    if generated.numel():
                        scores[batch, generated.unique()] -= profile.presence_penalty
                return scores

        class Progress(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                count = input_ids.shape[1] - prompt_len
                if count == 1 or count % 16 == 0:
                    update(stage=f"Enhancing prompt · {count} tokens", generated_tokens=count)
                return cancelled()

        processors = LogitsProcessorList([PresencePenalty()] if profile.presence_penalty else [])
        torch.manual_seed(seed)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=profile.max_new_tokens,
                do_sample=True, temperature=profile.temperature, top_p=profile.top_p, top_k=profile.top_k,
                logits_processor=processors, stopping_criteria=StoppingCriteriaList([Progress()]),
                pad_token_id=processor.tokenizer.eos_token_id)
        if cancelled():
            return None
        text = processor.tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
        _, answer = core.split_thinking(text)
        parsed = core.parse_answer(answer, profile)
        if not parsed["parse_ok"]:
            raise ValueError("The enhancer did not return a complete prompt. Your original direction is unchanged.")
        if len(parsed["positive_prompt"]) > 12000:
            raise ValueError("The enhanced prompt exceeded the image model's input limit")
        return dict(prompt=parsed["positive_prompt"], wh_ratio=parsed["wh_ratio"],
                    ratio_follow=parsed["ratio_follow"], task=task,
                    model="Qwen/" + root.name, quantization="nf4")
    finally:
        del output, inputs, model, processor
        gc.collect()
        torch.cuda.empty_cache()
