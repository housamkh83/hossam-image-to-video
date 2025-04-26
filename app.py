# -*- coding: utf-8 -*-
import gradio as gr
import torch
import os
from glob import glob
from pathlib import Path
from typing import Optional
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import uuid
import random
from huggingface_hub import hf_hub_download
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

# تعريف أقصى قيمة صحيحة 64 بت
max_64_bit_int = 9223372036854775807  # 2**63 - 1

# التحقق من توفر CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"

def download_progress_callback(progress_bar):
    def callback(evolution: float):
        progress_bar.update(evolution - progress_bar.n)
    return callback

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_model():
    try:
        progress_bar = tqdm(total=100, desc="Downloading model")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
            resume_download=True,  # Enable resuming interrupted downloads
            local_files_only=False,
            progress_callback=download_progress_callback(progress_bar)
        )
        progress_bar.close()
        return pipe
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# تهيئة نموذج المعالجة مع الإعدادات المحسنة
try:
    pipe = load_model()
    pipe.to(device)
except Exception as e:
    print(f"Failed to load model after retries: {str(e)}")
    raise

# تفعيل جميع التحسينات المتاحة
if hasattr(pipe, 'enable_model_cpu_offload'):
    pipe.enable_model_cpu_offload()
if hasattr(pipe, 'enable_vae_slicing'):
    pipe.enable_vae_slicing()
if hasattr(pipe, 'enable_vae_tiling'):
    pipe.enable_vae_tiling()

# تفعيل إعدادات CUDA المحسنة
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# تعديل دالة build_scene_prompt لتستقبل وصفاً واحداً
def build_scene_prompt(scene_description: str) -> str:
    """
    تحويل وصف المشهد إلى تنسيق مناسب
    """
    return scene_description.strip() if scene_description else "مشهد عام"

# تعديل دالة sample لتستقبل وصف المشهد كمعامل واحد
def sample(
    image: Image,
    seed: Optional[int] = 42,
    randomize_seed: bool = True,
    motion_bucket_id: int = 180,
    num_frames: int = 45,
    fps_id: int = 8,
    scene_description: str = "",  # وصف المشهد الإجباري
    version: str = "svd_xt",
    cond_aug: float = 0.03,
    decoding_t: int = 3,
    device: str = device,
    output_folder: str = "outputs",
    progress=gr.Progress(track_tqdm=True)
):
    if image.mode == "RGBA":
        image = image.convert("RGB")
        
    if randomize_seed:
        seed = random.randint(0, max_64_bit_int)
    generator = torch.manual_seed(seed)
    
    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

    # تحديث كيفية معالجة وصف المشهد
    scene_prompt = build_scene_prompt(scene_description)
    print(f"وصف المشهد: {scene_prompt}")  # للتصحيح

    # Fix the deprecation warning for torch.cuda.amp.autocast
    with torch.amp.autocast('cuda' if device == "cuda" else 'cpu'):
        frames = pipe(
            image,
            decode_chunk_size=decoding_t,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=cond_aug,
            num_frames=num_frames,
            num_inference_steps=25
        ).frames[0]
    
    export_to_video(frames, video_path, fps=fps_id)
    return video_path, seed

def resize_image(image, output_size=(1024, 1024)):
    if image is None:
        return None
        
    target_aspect = output_size[0] / output_size[1]
    image_aspect = image.width / image.height

    if image_aspect > target_aspect:
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image

custom_css = """
.gradio-logo, .footer, .gr-prose {
    display: none !important;
}
footer {
    display: none !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown('''# محول الصور إلى فيديو - v6
    تطوير: حسام فضل قدور
    
    التعليمات:
    1. قم برفع الصورة,وصف تفصيلي للمشهد (إجباري)
    2. يمكنك استخدام اللغة العربية أو الإنجليزية
    3. اضبط الخيارات المتقدمة حسب الحاجة
    4. ,ملاحظة مهمة بعد الخروج تحفظ في مجلد مباشر اتوماتيكي outputs انقر على زر توليد الفيديو ✨
    ''')
    
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="قم برفع الصورة", type="pil")
            generate_btn = gr.Button("✨ توليد الفيديو", variant="primary")
            clear_button = gr.Button("🗑️ مسح", variant="secondary")
        video = gr.Video()
        
    with gr.Accordion("🛠️ خيارات متقدمة", open=False):
        seed = gr.Slider(
            label="البذرة العشوائية",
            value=42,
            randomize=True,
            minimum=0,
            maximum=max_64_bit_int,
            step=1
        )
        randomize_seed = gr.Checkbox(label="بذرة عشوائية", value=True)
        motion_bucket_id = gr.Slider(
            label="شدة الحركة",
            info="التحكم في مقدار الحركة المضافة/المحذوفة",
            value=250,
            minimum=1,
            maximum=500
        )
        num_frames = gr.Slider(
            label="عدد الإطارات",
            info="إجمالي عدد الإطارات المراد توليدها",
            value=20,
            minimum=10,
            maximum=60
        )
        fps_id = gr.Slider(
            label="الإطارات في الثانية",
            info="التحكم في سرعة الفيديو (معدل إطارات أقل = فيديو أطول)",
            value=4,
            minimum=1,
            maximum=30
        )
      
    with gr.Accordion("🎬 وصف المشهد", open=True):
        scene_description = gr.Textbox(
            label="وصف المشهد",
            placeholder="اكتب وصفاً تفصيلياً للمشهد المطلوب. مثال: رجل يرتدي بدلة رسمية يمشي في شارع مزدحم تحت إضاءة المساء مع حركة كاميرا بطيئة من اليسار لليمين",
            info="وصف تفصيلي للمشهد (إجباري)",
            lines=3,
            value=""
        )

    image.upload(
        fn=resize_image,
        inputs=image,
        outputs=image,
        queue=False
    )
    
    def validate_inputs(image, description):
        if image is None:
            raise gr.Error("يرجى رفع صورة أولاً")
        if not description.strip():
            raise gr.Error("يرجى كتابة وصف للمشهد")
        return True

    generate_btn.click(
        fn=validate_inputs,
        inputs=[image, scene_description],
        outputs=None,
        api_name=None,
    ).success(
        fn=sample,
        inputs=[
            image, seed, randomize_seed, motion_bucket_id,
            num_frames, fps_id, scene_description
        ],
        outputs=[video, seed],
        api_name="video"
    )

    # إضافة حقوق الملكية
    gr.HTML(
        '<div class="custom-footer">جميع الحقوق محفوظة © 2024 تطوير: حسام فضل قدور</div>'
    )

    # Add this after creating the clear_button
    def clear_outputs():
        return None, None, ""  # إضافة مسح وصف المشهد

    clear_button.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[image, video, scene_description],
        queue=False
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=7861,
        show_api=False
    )