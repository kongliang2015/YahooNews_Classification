
# coding: utf-8
import sys
import csv
import jieba
import re
import MeCab


# In[ ]:

# reload(sys)
# sys.setdefaultencoding('utf-8')
csv.field_size_limit(sys.maxsize)


# In[ ]:

# 分词
def wakati(text,mod):
    
    if mod == 'jieba':
        # 用jieba分词
        return jieba.cut(text)
    elif mod == 'mecab':
        # 用mecab分词
        tagger = MeCab.Tagger("-Owakati")        
        return tagger.parse(text)


# In[ ]:

# 去除标点符号和特殊符号
def scan(line):
    string = re.sub("[+\:;?\"\.\!\/_,$%^*(+\"\']+|[+——！，“”·-▲▼▽•★ˇ<>「」。？、》《~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),line.decode("utf8")) 
    return string


# In[ ]:

# 读取CSV文件
def readDataFile(fileName,ptype):
    
    if ptype == 'train':
        # 训练数据4个字段
        COND = 4
    elif ptype == 'test':
        # 测试数据3个字段
        COND = 3
    
    with open(fileName,"rb") as csvfile:
        reader = csv.reader(csvfile,delimiter = '\t')
        # 读取内容
        # 用csv.reader 读取 line 已经是list了
        for line in reader:
            if len(line) == COND:
                # 标题分词                               
                seg_title = wakati(scan(line[1].strip()),'jieba')
                seg_title_return = " ".join(seg_title)
#                 print ",".join(seg_title)
                # 正文分词
                seg_content = wakati(scan(line[2].strip()),'jieba')
                seg_content_return = " ".join(seg_content)
#                 print ",".join(seg_content)
                if ptype == 'train':
                    # label = line[3]
                    yield seg_title_return,seg_content_return,line[3].strip()
                elif ptype == 'test':
                    # doc_id = line[0]
                    yield seg_title_return,seg_content_return,line[0].strip()


# In[ ]:

# 把分词以后的结果写入文件
def writeFile(outputfile,newline):
    
    fw = open(outputfile, 'ab')
    fw.write(newline.encode("utf-8"))
    fw.close()


# In[ ]:

def run(input_file,outfile_title,outfile_content,outfile_all,ptype):
    # 定义训练文件

    # 每次获取最大件数
    STOP_SIZE = 500000
    # 初始化ctr值
    ctr = 0
    
    for i in readDataFile(input_file,ptype):
        if ctr< STOP_SIZE:
            ctr +=1
            print "<%s>" % (ptype) + "第" + str(ctr)+ "件"
            #去除标点和特殊符号后的标题和内容取得
            title = i[0]
            content = i[1]
            # if ptype == 'train':label = i[2]
            # if ptype == 'test':doc_id = i[2]
            if ptype == 'train':
                new_title = '__label__' + i[2] + " " + title + "\n"
                new_content = '__label__' + i[2]+ " " + content + "\n"
                new_all = '__label__' + i[2] + " "+ title + " " + content+ "\n"
            elif ptype == 'test':
                new_title = i[2] + " " + title + "\n"
                new_content = i[2]+ " " + content + "\n"
                new_all = i[2] + " "+ title + " " + content+ "\n"
            
            #写入新的文件
            writeFile(outfile_title,new_title)
            writeFile(outfile_content,new_content)
            writeFile(outfile_all,new_all)
            
        else:
            break


# In[ ]:

# Main函数
def main():
    # 训练数据
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360/data/"
    input_train = dataPath + "train.tsv"
    train_outfile_title = dataPath + "train/train_output_title.tsv"
    train_outfile_content = dataPath + "train/train_output_content.tsv"
    
    train_outfile_all = dataPath + "train/train_all.tsv"
    
    # 测试数据
    input_test = dataPath + "evaluation_public.tsv"
    test_outfile_title = dataPath + "test/test_output_title.tsv"
    test_outfile_content = dataPath + "test/test_output_content.tsv"
    
    test_outfile_all = dataPath + "test/test_all.tsv"
    
    # 处理训练文件
#     run(input_train,train_outfile_title,train_outfile_content,train_outfile_all,'train')
    
    # 处理测试文件
    run(input_test,test_outfile_title,test_outfile_content,test_outfile_all,'test')
    


# In[ ]:

if __name__ == '__main__':
    main()

