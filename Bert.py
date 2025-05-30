from transformers import BertModel, BertTokenizer
import torch
//
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# 输入示例
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print("BERT 输出：", last_hidden_states.shape)  #[1, 12, 768]
