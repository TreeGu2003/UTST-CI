import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import defaultdict

class StyleSimilarityScorer:
    def __init__(self, style_word_files):
        """
        初始化评分器
        :param style_word_files: 字典，格式为 {'style1': 'path1.csv', 'style2': 'path2.csv'}
        """
        self.style_words = {}
        self.style_vectors = defaultdict(dict)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # 加载各个风格的词汇表
        for style, filepath in style_word_files.items():
            df = pd.read_csv(filepath)
            self.style_words[style] = set(df['word'].head(1000).tolist())  # 取前1000个词
            
        # 为每个风格创建参考文本（包含该风格的所有特征词）
        self.reference_texts = {
            style: ' '.join(words) 
            for style, words in self.style_words.items()
        }
        
        # 训练TF-IDF向量化器（在所有风格词汇上）
        all_words = []
        for words in self.style_words.values():
            all_words.extend(words)
        self.vectorizer.fit([' '.join(all_words)])
        
        # 为每个风格创建TF-IDF向量
        for style, text in self.reference_texts.items():
            self.style_vectors[style] = self.vectorizer.transform([text])

    def calculate_style_similarity(self, text, target_style):
        """
        计算文本与指定风格词汇库的相似度
        :param text: 要评分的文本
        :param target_style: 要比较的目标风格
        :return: 相似度分数 (0-1)
        """
        if target_style not in self.style_vectors:
            raise ValueError(f"未知的风格: {target_style}. 可用风格: {list(self.style_vectors.keys())}")
        
        # 向量化输入文本
        text_vector = self.vectorizer.transform([text])
        
        # 计算与目标风格的余弦相似度
        similarity = cosine_similarity(text_vector, self.style_vectors[target_style])[0][0]
        
        # 确保分数在0-1范围内（余弦相似度理论上已经是，但可能有浮点误差）
        return max(0.0, min(1.0, similarity))
    
    def calculate_style_probabilities(self, text, temperature=0.01):
        """
        改进版概率计算，通过温度系数保持差异
        :param temperature: 越小则差异越明显 (推荐0.01-0.5)
        """
        raw_scores = {
            style: self.calculate_style_similarity(text, style)
            for style in self.style_vectors.keys()
        }
        scores = np.array(list(raw_scores.values()))
        
        # 带温度系数的softmax
        scaled_scores = scores / temperature
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        return {
            style: float(prob)
            for style, prob in zip(raw_scores.keys(), probabilities)
        }

# 使用示例
if __name__ == "__main__":
    # 初始化评分器（假设你已经生成了之前的词汇文件）
    style_files = {
        'elementary': 'readability_style_differences/elementary_specific_words.csv',
        'middle': 'readability_style_differences/middle_specific_words.csv',
        'high': 'readability_style_differences/high_specific_words.csv',
        'college': 'readability_style_differences/college_specific_words.csv'
    }
    
    scorer = StyleSimilarityScorer(style_files)
    
    # 测试文本
    test_text = "Annette Miller, a member of Fit Nation, has a new motto for her life. She believes that there is a big difference between being inspired by someone and comparing ourselves to them. Miller says that when we compare ourselves to others, it makes it hard to see things clearly."
    
    print("原始相似度分数:")
    for style in style_files.keys():
        score = scorer.calculate_style_similarity(test_text, style)
        print(f"文本与 {style} 风格的相似度: {score:.4f}")
    
    print("\nSoftmax归一化后的概率分布:")
    probs = scorer.calculate_style_probabilities(test_text, 1)
    for style, prob in probs.items():
        print(f"文本属于 {style} 风格的概率: {prob:.4f}")
    
    # 验证概率和为1
    print(f"\n概率总和: {sum(probs.values()):.4f}")

    elementary_prob = probs['elementary']
    print(f"Elementary概率: {elementary_prob:.4f}")