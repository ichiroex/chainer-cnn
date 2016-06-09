# coding: utf-8
import numpy as np
from gensim.models import word2vec

def padding(document_list, max_len):
    
    new_document_list = []
    for doc in document_list:
        pad_line = ['<pad>' for i in range(max_len - len(doc))] #全ての文書の単語数を合わせる
        new_document_list.append(doc + pad_line)
    return new_document_list

def load_data(fname):

    model =  word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    target = [] #ラベル
    source = [] #文書ベクトル

    #文書リストを作成
    document_list = []
    for l in open(fname, 'r').readlines():
        sample = l.strip().split(' ',  1)
        label = sample[0]
        target.append(label) #ラベル
        document_list.append(sample[1].split()) #文書ごとの単語リスト
    
    max_len = 0
    rev_document_list = [] #未知語処理後のdocument list
    for doc in document_list:
        rev_doc = []
        for word in doc:
            try:
                word_vec = np.array(model[word]) #未知語の場合, KeyErrorが起きる
                rev_doc.append(word)
            except KeyError:
                rev_doc.append('<unk>') #未知語
        rev_document_list.append(rev_doc)
        #文書の最大長を求める(padding用)
        if len(rev_doc) > max_len:
            max_len = len(rev_doc)
    
    #文書長をpaddingにより合わせる
    rev_document_list = padding(rev_document_list, max_len)
    
    width = 0 #各単語の次元数
    #文書の特徴ベクトル化
    for doc in rev_document_list:
        doc_vec = []
        for word in doc:
            try:
                vec = model[word.decode('utf-8')]
            except KeyError:
                vec = model.seeded_vector(word)
            doc_vec.extend(vec)
            width = len(vec)
        source.append(doc_vec)

    dataset = {}
    dataset['target'] = np.array(target)    
    dataset['source'] = np.array(source)    

    return dataset, max_len, width

