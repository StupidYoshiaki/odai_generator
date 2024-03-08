from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from transformers import T5Tokenizer,AutoModelForCausalLM, AutoTokenizer
import torch

import pandas as pd
import os

# gpu使えるか確認
print(torch.cuda.is_available())


# モデルの読み込み
# "rinna/japanese-gpt2-medium" "rinna/japanese-gpt-1b" "rinna/japanese-gpt2-small"
model_name = "rinna/japanese-gpt2-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

train_data_path = "./data/train.txt"
valid_data_path = "./data/valid.txt"

#データセットの設定
train_dataset = TextDataset(
    tokenizer = tokenizer,
    file_path = train_data_path,
    block_size = 128 #文章の長さを揃える必要がある
)

#データセットの設定
validation_dataset = TextDataset(
    tokenizer = tokenizer,
    file_path = valid_data_path,
    block_size = 128 #文章の長さを揃える必要がある
)

#データの入力に関する設定
data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = False
)

# resultディレクトリのディレクトリ数を取得する
v_num = len(os.listdir("./result/gpt_odai")) + 1

# 訓練に関する設定
training_args = TrainingArguments(
    output_dir=f"./result/gpt_odai/v{v_num}",  # 関連ファイルを保存するパス
    per_device_train_batch_size=32, # 訓練時のバッチサイズ
    per_device_eval_batch_size=32, # 評価時のバッチサイズ
    learning_rate=1e-4, # 学習率
    lr_scheduler_type="linear", # 学習率スケジューラ
    warmup_ratio=0.1, # 学習率のウォームアップ
    num_train_epochs=5, # 訓練エポック数
    evaluation_strategy="epoch", # 評価タイミング
    save_strategy="epoch", # チェックポイントの保存タイミング
    logging_strategy="epoch", # ロギングのタイミング
    load_best_model_at_end=True, # 訓練後に検証セットで最良のモデルをロード
)

#トレーナーの設定
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer
)

trainer.train()