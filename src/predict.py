from transformers import T5Tokenizer,AutoModelForCausalLM
import torch

import numpy as np
import pandas as pd
import os
import sys



def getarate_sentences(seed_sentence, tokenizer, model):
    x = tokenizer.encode(seed_sentence, return_tensors="pt", 
    add_special_tokens=False)  # 入力
    x = x.cuda()  # GPU対応
    y = model.generate(x, #入力
                        min_length=5,  # 文章の最小長
                        max_length=100,  # 文章の最大長
                        do_sample=True,   # 次の単語を確率で選ぶ
                        top_k=50, # Top-Kサンプリング
                        top_p=0.95,  # Top-pサンプリング
                        temperature=1.0,  # 確率分布の調整
                        num_return_sequences=3,  # 生成する文章の数
                        pad_token_id=tokenizer.pad_token_id,  # パディングのトークンID
                        bos_token_id=tokenizer.bos_token_id,  # テキスト先頭のトークンID
                        eos_token_id=tokenizer.eos_token_id,  # テキスト終端のトークンID
                        # bad_word_ids=[[tokenizer.unk_token_id]]  # 生成が許可されないトークンID
                        )  
    generated_sentences = tokenizer.batch_decode(y, skip_special_tokens=True)  # 特殊トークンをスキップして文章に変換
    return generated_sentences


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # resultディレクトリのディレクトリ数を取得する
    v_num = len(os.listdir("./result/gpt_odai"))
    
    # gptの読み込み
    model_name = f"./result/gpt_odai/v{v_num}/checkpoint-14085"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    with open(f"./result/gpt_odai/v{v_num}/odai_generated.txt", "w") as f:
        for _ in range(10):
            seed_sentence = "お題:"
            generated_sentences = getarate_sentences(seed_sentence, tokenizer, model)
            for sentence in generated_sentences:
                f.write(sentence + "\n")
          
            
if __name__ == "__main__":
    main()