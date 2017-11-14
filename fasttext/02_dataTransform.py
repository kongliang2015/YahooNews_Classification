
# coding: utf-8

# In[20]:


import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# In[21]:

import numpy as np
from keras.preprocessing import sequence
import collections  #用来统计词频
import pickle


# In[22]:

def getLable(x):
    if x == 'NEGATIVE':
        lable = 0
    elif x== 'POSITIVE':
        lable = 1
    else:
        lable = -1
        print x
    return lable


# In[23]:

# 获取训练数据
def getTrainSet(inFile):
    # 训练集
    train_set=[]
    # 情感标签集
    target_set = []
    # 统计所有出现的词
    word_ctr = collections.Counter()
    # 评论的最大长度
    maxlen = 0
    len_ctr = collections.Counter()
    
    # 读入训练数据           
    f=open(inFile)
    lines=f.readlines()
    for line in lines:
        article = line.replace('\n','').split('\t')

        # 情感标签
        target_set.append(getLable(article[0]))

        # 内容
        content = article[1:]
        train_set.append(content)

        # 获得评论的最大长度
        if len(content) > maxlen:
            maxlen = len(content)

        # 统计各种长度的文章个数
        len_ctr[str(len(content))] += 1

        # 统计所有出现的词
        for w in content:
            word_ctr[w] += 1

    f.close()
        
    print('max_len ',maxlen)
    print('nb_words ', len(word_ctr))
#     print ('len_ctr ', len_ctr)
    return (target_set,train_set,word_ctr)


# In[24]:

# # 获取测试数据
def getTestSet(inFile):
    # 训练集
    train_set=[]
    # 情感标签集
    docid_set = []
    # 统计所有出现的词
    word_ctr = collections.Counter()
    
    # 读入训练数据
    f=open(inFile)
    lines=f.readlines()
    for line in lines:
        article = line.replace('\n','').split('\t')
        
        # 情感标签
        docid_set.append(article[0])
       
        # 内容
        content = article[1:]
        train_set.append(content)
        
        # 统计所有出现的词
        for w in content:
            word_ctr[w] += 1

    f.close()
    print('nb_words ', len(word_ctr))
    return (docid_set,train_set,word_ctr)


# In[25]:

# 把原始文本转化为由词汇表索引表示的矩阵
def dataTransform(inFile,outFile,maxfeat,seqlen):
    # 读入训练数据
    print "read file:",inFile[0]
    target_set,train_set,word_ctr = getTrainSet(inFile[0])
    
    # 读入测试数据
    print "read file:",inFile[1]
    docid_set,test_set,word_ctr_test = getTestSet(inFile[1])    
    
    # 实际标题的长度很多在20左右
    MAX_SENTENCE_LENGTH = seqlen
    
    # ('nb_words ', 236825)
    MAX_FEATURES = maxfeat
        
    # 合并词表 适用update()方法
    word_ctr.update(word_ctr_test)
    print "new word_ctr:",len(word_ctr)
    
    # 对于不在词汇表里的单词，把它们用伪单词 UNK 代替。 
    # 根据句子的最大长度 (max_lens)，我们可以统一句子的长度，把短句用 0 填充。
    # 接下来建立两个 lookup tables，分别是 word2index 和 index2word，用于单词和数字转换。 
    vocab_size = min(MAX_FEATURES, len(word_ctr)) + 2
    word2index = {x[0]: i+2 for i, x in enumerate(word_ctr.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v:k for k, v in word2index.items()}
    
#     np.save("./model/word2index.npy",word2index)
#     np.save("./model/index2word.npy",index2word)
    
    # 对训练数据做转换  
    X = np.empty(len(train_set),dtype=list)
    y = np.array([int(i) for i in target_set])
    
    i = 0
    for news in train_set:
        trs_news = []
        for w in news:
            if w in word2index:
                trs_news.append(word2index[w])
            else:
                trs_news.append(word2index['UNK'])
        X[i] = trs_news
        i += 1
    print "%s train files is transformed." % (i)
        
    
    # 对文字序列做补齐 ，补齐长度=最长的文章长度 ，补齐在最后，补齐用的词汇默认是词汇表index=0的词汇，也可通过value指定
    # 训练好的w2v词表的index = 0 对应的词汇是空格
    X = sequence.pad_sequences(X,maxlen=MAX_SENTENCE_LENGTH,padding='post')
    
    print "save train set matirx to np file..."
    
    np.save(outFile[0],np.column_stack([X,y]))
    
    ####################
    # 对测试数据做转换    
    # 生成一个<idx,doc_id>键值对
    tmp = [[idx,docid] for idx,docid in enumerate(docid_set)]
    # 序列化
    with open(outFile[2], 'wb') as f:
        pickle.dump(tmp, f)
    
    Xt = np.empty(len(test_set),dtype=list)
    yt = np.array([int(i[0]) for i in tmp])
    
    i = 0
    for news in test_set:
        trs_news = []
        for w in news:
            if w in word2index:
                trs_news.append(word2index[w])
            else:
                trs_news.append(word2index['UNK'])
        Xt[i] = trs_news
        i += 1
    print "%s test files is transformed." % (i)
        
    
    # 对文字序列做补齐 ，补齐长度=最长的文章长度 ，补齐在最后，补齐用的词汇默认是词汇表index=0的词汇，也可通过value指定
    # 训练好的w2v词表的index = 0 对应的词汇是空格
    Xt = sequence.pad_sequences(Xt,maxlen=MAX_SENTENCE_LENGTH,padding='post')
    
    print "save test set matirx to np file..."
    
    np.save(outFile[1],np.column_stack([Xt,yt]))
    


# In[26]:

def run(trainFile,matFile):
    
    # 定义词典长度
    FEATURES = 2000000
    # 定义序列长度
#     SENTENCE_LENGTH = 20
    SENTENCE_LENGTH = 200
    
    # 把训练文本转化为矩阵
    dataTransform(trainFile,matFile,FEATURES,SENTENCE_LENGTH)
    


# In[ ]:

def main():
    
    # 定义文件路径
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360/data/"
    mdlPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360/model/"
    
    # 定义训练文件名
    trainFile_title = dataPath + "train/train_output_title.tsv"
    # 定义输出文件名
    train_matFile_title = mdlPath + "train_matrix_title.npy"    
    
    trainFile_content = dataPath + "train/train_output_content.tsv"
    train_matFile_content = mdlPath + "train_matrix_content.npy"
        
    # 定义测试文件名
    testFile_title = dataPath + "test/test_output_title.tsv"
    # 定义输出文件名
    test_matFile_title = mdlPath + "test_matrix_title.npy"
    
    testFile_content = dataPath + "test/test_output_content.tsv"    
    test_matFile_content = mdlPath + "test_matrix_content.npy"
    
    # <idx,doc_id>键值对文件
    train_pkl = mdlPath + "train_pkl.txt"
        
    # 把所有文件的标题读入，建立字典，把原标题转化为矩阵
#     run([trainFile_title,testFile_title],[train_matFile_title,test_matFile_title,train_pkl])
    
    # 转换内容文件
    run([trainFile_content,testFile_content],[train_matFile_content,test_matFile_content,train_pkl])
    


# In[ ]:

if __name__ == '__main__':
    main()

