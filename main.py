import os
from pdf2image import convert_from_path
import pytesseract
from openai import OpenAI
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import logging

# تنظیم لاگ‌گذاری
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# بارگذاری متغیرهای محیطی
load_dotenv()

# تنظیم مسیر Tesseract برای لینوکس یا ویندوز
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")

# کلید API و مسیرهای فایل
API_KEY = os.getenv("OPENAI_API_KEY")
PDF_FILE_PATH = os.getenv("PDF_FILE_PATH", "input.pdf")
POPPLER_PATH = os.getenv("POPPLER_PATH", "/usr/bin")  # مسیر پیش‌فرض در لینوکس
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp")  # ذخیره موقت در لینوکس

# بررسی وجود متغیرهای محیطی
if not API_KEY:
    raise ValueError("کلید API OpenAI در متغیرهای محیطی تنظیم نشده است.")
if not os.path.exists(PDF_FILE_PATH):
    raise FileNotFoundError(f"فایل PDF در مسیر {PDF_FILE_PATH} یافت نشد.")

# مقداردهی اولیه کلاینت OpenAI
client = OpenAI(api_key=API_KEY)

# پیش‌پردازش تصویر برای بهبود OCR
def preprocess_image(image):
    try:
        image = image.convert("L")  # تبدیل به خاکستری
        image = image.filter(ImageFilter.SHARPEN)  # افزایش وضوح
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # تنظیم کنتراست
        open_cv_image = np.array(image)
        _, thresh = cv2.threshold(open_cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((1, 1), np.uint8)
        processed_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return Image.fromarray(processed_image)
    except Exception as e:
        logger.error(f"خطا در پیش‌پردازش تصویر: {e}")
        return image  # در صورت خطا، تصویر اصلی را برگردان

# استخراج متن از یک صفحه
def process_page(image, page_number):
    try:
        logger.info(f"پردازش صفحه {page_number}...")
        processed_image = preprocess_image(image)
        text = pytesseract.image_to_string(processed_image, lang="fas", config="--psm 6")
        return text + "\n\n"
    except Exception as e:
        logger.error(f"خطا در OCR صفحه {page_number}: {e}")
        return ""

# استخراج متن از PDF
def extract_text_from_pdf(pdf_path, poppler_path):
    try:
        logger.info("تبدیل صفحات PDF به تصاویر...")
        # محدود کردن حافظه با تنظیم DPI و پردازش دسته‌ای
        images = convert_from_path(pdf_path, poppler_path=poppler_path, dpi=200, thread_count=4)
        full_text = ""

        # پردازش موازی صفحات
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(process_page, images, range(1, len(images) + 1))
            full_text = "".join(results)

        return full_text
    except Exception as e:
        logger.error(f"خطا در استخراج متن از PDF: {e}")
        raise

# خلاصه‌سازی متن با استفاده از OpenAI
def summarize_text(text):
    try:
        logger.info("خلاصه‌سازی متن...")
        # تقسیم متن به تکه‌های 1500 کاراکتری برای مدیریت بهتر محدودیت‌های API
        chunks = [text[i:i + 1500] for i in range(0, len(text), 1500)]
        summarized_text = ""

        for i, chunk in enumerate(chunks):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "متن زیر را به زبان فارسی خلاصه کن. از منطق وزن‌دهی به تیترها و جداسازی بخش‌های مهم استفاده کن."},
                        {"role": "user", "content": chunk}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                summarized_text += response.choices[0].message.content + "\n\n"
            except Exception as e:
                logger.error(f"خطا در خلاصه‌سازی تکه {i+1}: {e}")
                continue

        return summarized_text
    except Exception as e:
        logger.error(f"خطا در خلاصه‌سازی متن: {e}")
        return text  # در صورت خطا، متن اصلی را برگردان

# ذخیره متن در فایل
def save_to_file(text, filename):
    try:
        output_path = os.path.join(OUTPUT_DIR, filename)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"فایل در مسیر {output_path} ذخیره شد.")
    except Exception as e:
        logger.error(f"خطا در ذخیره فایل: {e}")
        raise

# اجرای اصلی
if __name__ == "__main__":
    try:
        # استخراج متن
        extracted_text = extract_text_from_pdf(PDF_FILE_PATH, POPPLER_PATH)
        save_to_file(extracted_text, "extracted_text.txt")

        # خلاصه‌سازی متن
        summarized_text = summarize_text(extracted_text)
        save_to_file(summarized_text, "summarized_text.txt")

    except Exception as e:
        logger.error(f"خطای کلی در اجرا: {e}")
        exit(1)
