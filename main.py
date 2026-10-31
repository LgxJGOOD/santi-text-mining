import os
import re
import jieba
from collections import Counter
from gensim.models import Word2Vec

# =====================
# 文件路径
# =====================
novel_path = "三体.txt"
stopwords_path = "stopwords.txt"
output_cut_path = "三体_分词结果.txt"
chapter_dir = "三体"


# =====================
# 1. 读取停用词
# =====================
def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f])


stopwords = load_stopwords(stopwords_path)


# =====================
# 2. 分词 + 去停用词
# =====================
def process_text(text):
    words = jieba.lcut(text)
    words = [w for w in words if w.strip() and w not in stopwords]
    return words


with open(novel_path, "r", encoding="utf-8") as f:
    content = f.read()

words = process_text(content)

# 保存分词结果
with open(output_cut_path, "w", encoding="utf-8") as f:
    f.write(" ".join(words))

print("✅ 分词完成")

# =====================
# 3. 按章节拆分
# =====================
# 正则匹配章节
pattern = re.compile(r"(第[\d一二三四五六七八九十百千]+章\s*.*)")

lines = content.split("\n")

chapters = []
current_title = None
current_content = []

for line in lines:
    line = line.strip()
    if not line:
        continue

    match = re.match(r"第[\d一二三四五六七八九十百千]+章", line)

    if match:
        # 保存上一章
        if current_title:
            chapters.append((current_title, "\n".join(current_content)))

        # 提取章节号 + 标题
        parts = line.split()
        chapter_num = parts[0]
        title = "".join(parts[1:]).strip()
        current_title = f"{chapter_num}_{title}"
        current_content = []
    else:
        current_content.append(line)

# 最后一章
if current_title:
    chapters.append((current_title, "\n".join(current_content)))

# 创建目录
if not os.path.exists(chapter_dir):
    os.makedirs(chapter_dir)

# 保存章节
for title, text in chapters:
    file_path = os.path.join(chapter_dir, f"{title}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

print("✅ 章节拆分完成")

# =====================
# 4. 每章关键词统计
# =====================
keywords_output = "章节关键词统计.txt"

with open(keywords_output, "w", encoding="utf-8") as out:
    for title, text in chapters:
        words = process_text(text)
        counter = Counter(words)
        top10 = counter.most_common(10)

        out.write(f"{title}\n")
        for word, freq in top10:
            out.write(f"{word}: {freq}\n")
        out.write("\n")

print("✅ 关键词统计完成")

# =====================
# 5. 训练 Word2Vec
# =====================
# 每章作为一句
sentences = []

for _, text in chapters:
    words = process_text(text)
    sentences.append(words)

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# 查询与“三体”最相关的词
try:
    similar_words = model.wv.most_similar("三体", topn=10)
    print("\n🔍 与“三体”最相关的词：")
    for word, score in similar_words[:5]:
        print(word, score)
except KeyError:
    print("⚠️ 词表中没有“三体”，可能被过滤了（min_count 太大）")

print("✅ Word2Vec 训练完成")