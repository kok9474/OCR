import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from transformers import AutoTokenizer, AutoModelForVision2Seq, TrOCRProcessor

# 1. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델, 토크나이저, processor 로드

tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-printed")
model = AutoModelForVision2Seq.from_pretrained("./model_plus").to(device)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

# 3. 이미지 불러오기 및 전처리
image_path = "./car_number_image_dataset/Validation/OCR_valid_image/01고8385.jpg"  # 테스트할 이미지 경로
image = Image.open(image_path).convert("RGB")

# 해상도 보정 및 명도/선명도 향상 (선택사항)
image = ImageOps.pad(image, (384, 384), method=Image.BICUBIC, color=(255, 255, 255))
image = ImageEnhance.Contrast(image).enhance(1.5)
image = image.filter(ImageFilter.SHARPEN)

# 4. Processor로 전처리
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# 5. 모델 추론
model.eval()
with torch.no_grad():
    generated_ids = model.generate(pixel_values, max_length=64)

# 6. 디코딩 및 후처리
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
print("Token list:", tokens)
#cleaned_text = generated_text.replace("##", "")  # 서브워드 제거

# 7. 출력
print("Raw Output:", generated_text)
print("model.config.eos_token_id:", model.config.eos_token_id)
print("tokenizer.eos_token:", tokenizer.eos_token)
print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
print("토큰화 확인 :", tokenizer.tokenize("01가1234"))
