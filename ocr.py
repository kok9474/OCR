import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from PIL import ImageEnhance, ImageFilter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoTokenizer, AutoModelForVision2Seq
from transformers import pipeline
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
from transformers import default_data_collator
import numpy as np
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

FULL_TRAINING=True

# GPU 확인
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [ 모델 불러오기 ]
# tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")
# tokenizer.eos_token = tokenizer.sep_token
# tokenizer.eos_token_id = tokenizer.sep_token_id

tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-printed")
model = AutoModelForVision2Seq.from_pretrained("./model")
model.to(device)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

# [ 모델 파라미터 설정 ]
# set special tokens used for creating the decoder_input_ids from the labels - 디코더 입력 관련 설정
model.config.decoder_start_token_id = tokenizer.cls_token_id # 디코더의 시작 토큰 정의 (텍스트 생성할 때 첫 입력으로 시작 토큰 필요)
model.config.pad_token_id = tokenizer.pad_token_id # padding에 사용하는 토큰 ID (손실 계산 시 padding은 무시)
model.config.vocab_size = model.config.decoder.vocab_size # 모델 출력 vocabulary 크기를 디코더의 vocabulary 크기로 맞추기
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 10 # 생성할 텍스트의 최대길이 설정
model.config.early_stopping = False # beam search가 모든 후보에서 eos_token_id를 생성하면 즉시 종료하도록 설정
model.config.no_repeat_ngram_size = 3 # 중복된 n-gram이 생성되지 않도록 제약
model.config.length_penalty = 2.0 # beam search 시 긴 문장을 더 불리하게 점수화 (beam search시 긴 문장을 더 불리하게 점수화)
model.config.num_beams = 4 # beam search 시 고려할 후보 (4개 beam 유지, 값이 클수록 다양한 문장을 탐색, 속도 느려짐)


# 데이터셋
class OCRDataset(Dataset):
    def __init__(self, dataset_dir, df, processor, tokenizer, max_target_length=32):
        self.dataset_dir = dataset_dir
        self.df = df
        self.processor = processor
        self.processor.image_processor.do_resize = False
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    # 이미지 파일 열기(processor 전처리), 텍스트(tokenizer 인코딩), 라벨에서 padding token을 -100으로 바꾸기(loss 무시용)
    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.dataset_dir + file_name).convert("RGB")
        image = ImageOps.pad(image, (384, 384), method=Image.BICUBIC, color=(255, 255, 255))
        # 대비 증가
        image = ImageEnhance.Contrast(image).enhance(1.2)
        # 선명하게
        image = image.filter(ImageFilter.SHARPEN)
        #image = ImageOps.pad(image, (384, 384), method=Image.BILINEAR, color=(255,255,255))
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text      
        labels = self.tokenizer(text, 
                                padding="max_length", 
                                #stride=32,
                                truncation=True,
                                max_length=self.max_target_length,
                                ).input_ids
        
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


train_img_dir = "./output_dataset/train/images/"
val_img_dir = "./output_dataset/val/images/"

def load_label_from_folder(json_dir):
    data = []
    
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            json_path = os.path.join(json_dir, file_name)
            with open(json_path, "r", encoding="utf-8") as f:
                label = json.load(f)
                data.append({
                    "file_name": label["file_name"],  # 예: "01가0865.jpg" / imagePath
                    "text": label["label"]             # 예: "01가0865" / value
                })
    
    return pd.DataFrame(data)

# 학습/검증 라벨 폴더 경로
train_label_dir = "./output_dataset/train/labels"
val_label_dir = "./output_dataset/val/labels"

# DataFrame으로 변환
train_df = load_label_from_folder(train_label_dir)
val_df = load_label_from_folder(val_label_dir)

# Dataset 생성
train_dataset = OCRDataset(
    dataset_dir=train_img_dir,  # 이미지 폴더
    df=train_df,
    tokenizer=tokenizer,
    processor=processor,
    max_target_length=10
)

eval_dataset = OCRDataset(
    dataset_dir=val_img_dir,
    df=val_df,
    tokenizer=tokenizer,
    processor=processor,
    max_target_length=10
)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset)) 

# training

# 학습 파라미터 지정
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="steps",
    load_best_model_at_end=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=7,
    fp16=True,
    learning_rate=4e-5,
    output_dir="./",
    logging_dir="./logs",
    logging_steps=100,
    save_steps=2000,
    eval_steps=1000,
    logging_strategy="steps",
    save_total_limit=2
)

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

# 평가 지표 정의
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

print_gpu_utilization()
result = trainer.train() # train()이 끝나면 result로 로그 정보 저장
# result = trainer.train(resume_from_checkpoint="./checkpoint-5000")

#from pprint import pprint
#pprint(trainer.state.log_history)

# 로그 기록에서 loss, cer, wer 정보 추출
steps = []
losses = []
cers = []
wers = []

for obj in trainer.state.log_history:
    if 'step' in obj:
        step = obj['step']
        steps.append(step)
        losses.append(obj.get('loss'))

        # CER과 WER은 평가 스텝에서만 존재하므로 조건 추가
        cer = obj.get('eval_cer')
        wer = obj.get('eval_wer')
        
        # 없는 경우 NaN으로 처리 (그래프는 NaN은 건너뜀)
        cers.append(cer if cer is not None else float('nan'))
        wers.append(wer if wer is not None else float('nan'))

# 그래프 그리기
plt.figure(figsize=(14, 5))

# Loss
plt.subplot(1, 3, 1)
plt.plot(steps, losses, label='Loss', color='blue')
plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.title('Training Loss over Steps')
plt.grid(True)

# CER
cer_steps = [s for s, c in zip(steps, cers) if c == c]  # c == c는 NaN 판별
cer_values = [c for c in cers if c == c]

plt.subplot(1, 3, 2)
plt.plot(cer_steps, cer_values, label='CER', color='orange')
plt.xlabel('Step')
plt.ylabel('CER')
plt.title('Character Error Rate')
plt.grid(True)

# WER
wer_steps = [s for s, w in zip(steps, wers) if w == w]
wer_values = [w for w in wers if w == w]

plt.subplot(1, 3, 3)
plt.plot(wer_steps, wer_values, label='WER', color='green')
plt.xlabel('Step')
plt.ylabel('WER')
plt.title('Word Error Rate')
plt.grid(True)

plt.tight_layout()

# 그래프 저장
save_path = "training_metrics.png"
plt.savefig(save_path, dpi=300)  # 고해상도 저장
print("training_metrics.png 파일로 그래프 저장")

trainer.save_model(output_dir="./model_plus") # 파인튜닝된 모델을 디스크에 저장

# evaluation
model.eval()
with torch.no_grad():
    eval_result = trainer.evaluate(eval_dataset, max_length=12)    

print(eval_result)

# 로그에서 평가 손실 추출
eval_steps = []
eval_losses = []

for entry in trainer.state.log_history:
    if "eval_loss" in entry and "step" in entry:
        eval_steps.append(entry["step"])
        eval_losses.append(entry["eval_loss"])

# 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(eval_steps, eval_losses, marker='o', color='red', label="Validation Loss")
plt.xlabel("Step")
plt.ylabel("Eval Loss")
plt.title("Validation Loss over Steps")
plt.grid(True)
plt.legend()

# 이미지 저장
plt.savefig("validation_loss.png", dpi=300)
plt.close()

print("validation_loss.png 파일로 그래프 저장")

# 이미지 하나 예측
sample_img_path = './car_number_image_dataset/Validation/OCR_valid_image/01고8385.jpg'
image = Image.open(sample_img_path).convert("RGB")

# 학습과 동일한 전처리 적용
image = ImageOps.pad(image, (384, 384), method=Image.BICUBIC, color=(255, 255, 255))
image = ImageEnhance.Contrast(image).enhance(1.5)
image = image.filter(ImageFilter.SHARPEN)

# processor 적용
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# 추론
with torch.no_grad():
    generated_ids = model.generate(pixel_values, 
                                   max_length=10, 
                                   eos_token_id=model.config.eos_token_id,
                                   pad_token_id=tokenizer.pad_token_id
                                   )

# 디코딩
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 결과 출력
print(generated_ids)
print("예측 결과:", generated_text)