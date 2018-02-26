# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:18:56 2018

@author: ASUS
"""

import jieba as jb
# 中文分词
text = "我爱北京天安门，天安门上太阳升。"
seg_list = jb.cut(text, cut_all=True)
print("Full Mode:" + "/".join(seg_list))
'''
Full Mode:我/爱/北京/天安/天安门///天安/天安门/门上/太阳/太阳升//
'''

seg_list = jb.cut(text, cut_all=False)
print("Default Mode:" + "/".join(seg_list))
'''
Default Mode:我/爱/北京/天安门/，/天安门/上/太阳升/。
'''

seg_list = jb.cut_for_search(text, HMM=True)
print("Search Mode:" + "/".join(seg_list))
'''
Search Mode:我/爱/北京/天安/天安门/，/天安/天安门/上/太阳/太阳升/。
'''

# 关键词提取
import jieba.analyse as ja
# TF-IDF
keyword = ja.extract_tags(sentence, topK=20, withWeight=False, allowPOS=(), withFlag=False)
# TextRank
keyword = ja.textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False)

# 词性标注
import jieba.posseg as jp
text = "我爱北京天安门，天安门上太阳升。"
words = jp.cut(text)
for word, flag in words:
    print("%s, %s" % (word, flag))
    
# 中文分词工具jieba中的词性类型
# https://www.cnblogs.com/adienhsuan/p/5674033.html
