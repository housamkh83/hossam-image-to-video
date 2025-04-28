# 🎬 hossam-image-to-video: تحويل الصور إلى فيديو بالذكاء الاصطناعي

<p align="center">
  <img src="assets/logo.png" alt="شعار KH" width="100"/> <!-- ضع الشعار في مجلد assets -->
</p>

<p align="center">
  <strong>✨ أداة لتحريك صورك وإضافة الحياة إليها ✨</strong>
</p>
<p align="center">
  <strong>🏆 من تطوير: حسام فضل قدور 🏆</strong>
</p>

<p align="center">
  <a href="رابط-HuggingFace-Space-إذا-وجد" target="_blank" style="margin: 0 5px;">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face%20Space-تجربة%20مباشرة-blue?style=flat-square" alt="Hugging Face Space"/>
  </a>
  <img src="https://img.shields.io/github/license/housamkh83/hossam-image-to-video?style=flat-square" alt="GitHub License">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python" alt="Python Version">
   <img src="https://img.shields.io/badge/Model-Stable%20Video%20Diffusion-orange?style=flat-square" alt="Model SVD">
   <img src="https://img.shields.io/badge/UI-Gradio-orange?style=flat-square" alt="UI Gradio">
</p>

---

## 🌟 نظرة عامة

هذا المشروع مفتوح المصدر يوفر أداة سهلة الاستخدام لتحويل الصور الثابتة إلى مقاطع فيديو قصيرة متحركة باستخدام نموذج **Stable Video Diffusion (SVD)** القوي من Stability AI (`stabilityai/stable-video-diffusion-img2vid-xt`). تم بناء الواجهة باستخدام **Gradio** وهي متاحة باللغة العربية.

يهدف هذا المشروع إلى تمكين المستخدمين من إضافة لمسة ديناميكية وحيوية لصورهم بسهولة، سواء كانوا فنانين، مصممين، أو مجرد هواة يرغبون في تجربة قدرات الذكاء الاصطناعي في توليد الفيديو.

تم تطوير هذه الأداة كجزء من **[منصة حسام للذكاء الاصطناعي (hossam-ai-suite)](https://github.com/housamkh83/hossam-ai-suite)**.

## ✨ أهم الميزات

*   **تحريك الصور:** تحويل أي صورة ثابتة إلى فيديو قصير متحرك.
*   **نموذج SVD:** يعتمد على نموذج `stable-video-diffusion-img2vid-xt` القوي.
*   **واجهة Gradio عربية:** واجهة بسيطة ومباشرة لرفع الصور وضبط الإعدادات.
*   **تغيير حجم تلقائي:** تقوم الأداة بتغيير حجم الصورة المدخلة تلقائياً إلى الأبعاد المناسبة للنموذج (1024x1024) مع الحفاظ على نسبة العرض إلى الارتفاع وقص الأطراف الزائدة.
*   **تحكم في الحركة والإخراج:**
    *   **شدة الحركة (Motion Bucket ID):** التحكم في مقدار "الحركة" التي يضيفها النموذج للصورة.
    *   **عدد الإطارات (Number of Frames):** تحديد طول الفيديو الناتج.
    *   **الإطارات في الثانية (FPS):** التحكم في سرعة تشغيل الفيديو (معدل إطارات أقل يعني فيديو أبطأ وأطول لنفس عدد الإطارات).
    *   **البذرة (Seed):** التحكم في عشوائية التوليد للحصول على نتائج مختلفة أو إعادة إنتاج نفس النتيجة.
*   **تحسينات الأداء والذاكرة:** يتضمن الكود تحسينات تلقائية لاستخدام `float16` على الـ GPU وتفعيل تقنيات مثل `enable_model_cpu_offload`, `enable_vae_slicing`, `enable_vae_tiling` لمحاولة العمل على أجهزة ذات ذاكرة VRAM محدودة.
*   **حفظ منظم:** يتم حفظ الفيديوهات الناتجة (MP4) في مجلد `outputs` مع ترقيم تسلسلي تلقائي (مثل `000000.mp4`, `000001.mp4`).

## ⚠️ هام: متطلبات النظام والقيود

*   **ذاكرة GPU (VRAM) عالية:** نماذج توليد الفيديو مثل SVD **تستهلك كمية كبيرة جداً من ذاكرة الـ GPU**. يوصى بشدة باستخدام بطاقة NVIDIA GPU بذاكرة **16GB VRAM أو أكثر** للحصول على أفضل أداء وتجنب أخطاء نفاد الذاكرة، خاصة عند توليد عدد إطارات كبير. قد تعمل الأداة على بطاقات بذاكرة أقل (مثل 12GB أو حتى 8GB) مع تفعيل تحسينات الذاكرة وتقليل عدد الإطارات، لكن الأداء سيكون أبطأ وقد تواجه مشاكل.
*   **FFmpeg:** هذه الأداة **ضرورية** لعملية استخراج الإطارات الأولية (إذا لزم الأمر في المستقبل) ولعملية تجميع الإطارات النهائية في ملف فيديو MP4. يجب تثبيت `ffmpeg` على نظامك وإضافته إلى متغيرات البيئة (PATH) لكي يتمكن الكود من استدعائه.
*   **وصف المشهد:** النسخة الحالية من الكود تتضمن حقلاً لوصف المشهد في الواجهة، لكن نموذج SVD في مكتبة `diffusers` لا يستخدم هذا الوصف النصي حالياً لتوجيه عملية التوليد.

## 🛠️ المتطلبات

*   **Python:** إصدار 3.9 أو أحدث.
*   **Git:** لتنزيل الكود.
*   **Pip:** لتثبيت المكتبات.
*   **FFmpeg:** يجب تثبيته على النظام وإضافته إلى PATH. (ابحث عن "install ffmpeg windows/linux/mac" للحصول على إرشادات).
*   **(موصى به بشدة):** NVIDIA GPU بذاكرة VRAM عالية (16GB+).
*   **(مطلوب لمستخدمي GPU):** CUDA Toolkit و cuDNN متوافقان مع إصدار PyTorch.

## ⚙️ التثبيت

1.  **تثبيت FFmpeg:** تأكد من تثبيت FFmpeg على نظامك أولاً.
2.  **نسخ المستودع:**
    ```bash
    git clone https://github.com/housamkh83/hossam-image-to-video.git
    cd hossam-image-to-video
    ```
3.  **(موصى به) إنشاء وتفعيل بيئة افتراضية:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux / macOS
    source venv/bin/activate
    ```
4.  **تحديث pip:**
    ```bash
    python -m pip install --upgrade pip
    ```
5.  **تثبيت PyTorch (هام: اختر الأمر المناسب لـ CUDA لديك):**
    ```bash
    # --- لمستخدمي NVIDIA GPU مع CUDA 11.8 ---
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # --- لمستخدمي NVIDIA GPU مع CUDA 12.1 ---
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # --- لمستخدمي CPU فقط (الأداء سيكون بطيئاً جداً وغير عملي غالباً) ---
    # pip install torch torchvision torchaudio
    ```
    *(تأكد من نجاح هذه الخطوة قبل المتابعة)*

6.  **تثبيت باقي المتطلبات:** (تأكد من وجود ملف `requirements.txt` صحيح في المستودع)
    ```bash
    pip install -r requirements.txt
    ```
    *(سيقوم بتثبيت Gradio, Diffusers, Transformers, Pillow, إلخ)*

## 🚀 التشغيل

1.  **تأكد من أنك داخل البيئة الافتراضية (`venv`) التي تم تثبيت المكتبات فيها.**
2.  **شغل التطبيق باستخدام:**
    ```bash
    python app.py [خيارات إضافية]
    ```
    *(افترضنا أن الكود تم حفظه باسم `app.py`)*
3.  **خيارات التشغيل (سطر الأوامر لـ `app.py`):**
    *   `--port <رقم>`: لتحديد منفذ مختلف (الافتراضي 7861 في الكود).
    *   `--share`: لتفعيل رابط Gradio العام.
    *   `--debug`: لتفعيل وضع التصحيح.

## 💻 الاستخدام

1.  بعد تشغيل التطبيق، افتح الرابط المحلي (عادةً `http://localhost:7861`) في متصفحك.
2.  **ارفع الصورة:** اضغط على منطقة رفع الصورة واختر الصورة التي تريد تحريكها. سيتم تغيير حجمها تلقائياً.
3.  **(اختياري حالياً) أدخل وصف المشهد:** يمكنك كتابة وصف للمشهد (لا يؤثر على النموذج حالياً).
4.  **(اختياري) اضبط الخيارات المتقدمة:** عدّل شدة الحركة، عدد الإطارات، سرعة الفيديو (FPS)، أو البذرة حسب رغبتك.
5.  **اضغط "توليد الفيديو"**.
6.  **انتظر:** عملية توليد الفيديو تستهلك وقتاً وموارد كبيرة، خاصة على الأجهزة ذات الذاكرة المحدودة. راقب شريط التقدم.
7.  **شاهد النتيجة:** سيظهر الفيديو الناتج في منطقة الإخراج عند اكتمال العملية. سيتم حفظه أيضاً في مجلد `outputs`.

## 📄 الترخيص

هذا المشروع مرخص تحت ترخيص **MIT**. انظر ملف [LICENSE](LICENSE).

## 🙏 شكر وتقدير

*   لفريق Stability AI على تطوير نموذج Stable Video Diffusion.
*   لفريق عمل مكتبات Diffusers, Transformers, Gradio من Hugging Face.
*   لمجتمع المصادر المفتوحة.

---

**تطوير: حسام فضل قدور** ([Hugging Face](https://huggingface.co/hussamkh83) | [AI Kit 8 Blog](https://aikit8.blogspot.com/))