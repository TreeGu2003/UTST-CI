import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

# 从JSONL文件加载数据
data = []
with open('combined_summary_prompt_parallel.json', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line.strip())
        data.append({
            'text': obj['output_text'],
            'style': obj['output_level']
        })

# 创建DataFrame
data = pd.DataFrame(data)

# 定义四种风格
styles = ['elementary', 'middle', 'high', 'college']

# 为每种风格准备文本
style_texts = {}
for style in styles:
    style_texts[style] = data[data['style'] == style]['text'].tolist()

# 合并所有文本用于TF-IDF向量化
all_texts = []
text_lengths = []
for style in styles:
    texts = style_texts[style]
    all_texts.extend(texts)
    text_lengths.append(len(texts))

# TF-IDF 向量化（增加max_features以获取更多词汇）
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_texts)
feature_names = vectorizer.get_feature_names_out()

# 计算每种风格的平均TF-IDF分数
style_tfidf = {}
start_idx = 0
for i, style in enumerate(styles):
    end_idx = start_idx + text_lengths[i]
    style_tfidf[style] = tfidf_matrix[start_idx:end_idx].mean(axis=0).A1
    start_idx = end_idx

# 创建输出目录
output_dir = 'style_differences'
os.makedirs(output_dir, exist_ok=True)

# 对每种风格，计算其与其他所有风格的平均差异
for target_style in styles:
    # 获取其他所有风格
    other_styles = [s for s in styles if s != target_style]
    
    # 计算其他风格的平均TF-IDF
    other_avg = sum(style_tfidf[s] for s in other_styles) / len(other_styles)
    
    # 计算目标风格与其他风格的平均差异
    diff = style_tfidf[target_style] - other_avg
    
    # 创建DataFrame并排序
    top_diff_words = pd.DataFrame({
        'word': feature_names,
        'score': diff
    }).sort_values(by='score', ascending=False)
    
    # 取前1000个词
    top_1000 = top_diff_words.head(1000)
    
    # 保存到文件
    filename = f"{output_dir}/{target_style}_specific_words.csv"
    top_1000.to_csv(filename, index=False)
    print(f"已保存 {target_style} 的1000个特征词到 {filename}")
    
    # 打印前20个词作为示例
    print(f"\n{target_style} 最突出的20个词:")
    print(top_1000.head(20))
    print("-" * 50)

print("\n所有风格的特征词已保存完毕！")