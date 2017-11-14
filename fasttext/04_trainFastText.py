# -*- encoding: utf-8 -*-
import logging
import fasttext

# reload(sys)
# sys.setdefaultencoding('utf-8')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

def trainSkipgram(inFile,mdlfile):
    # Skipgram model
    model = fasttext.skipgram(input_file=inFile,\
                              output=mdlfile,\
                              lr=0.1,\
                              dim=200,\
                              epoch=10,\
                              word_ngrams=3,\
                              bucket=5000000)
#     print "Skipgram model words:",model.words # list of words in dictionary

def trainCBOW(inFile,mdlfile):
    # CBOW model
    model = fasttext.cbow(input_file=inFile,\
                              output=mdlfile,\
                              lr=0.1,\
                              dim=200,\
                              epoch=10,\
                              word_ngrams=3,\
                              bucket=5000000)
#     print "CBOW model words:",model.words # list of words in dictionary

def trainCLF(inFile,model,vec):
    #训练模型
    classifier = fasttext.supervised(input_file=inFile,\
                                     output=model,\
                                     dim = 200, \
                                     word_ngrams = 2,\
                                     epoch = 10,\
                                     pretrained_vectors = vec)


def test(inFile, model):
    # load训练好的模型
    classifier = fasttext.load_model(model)

    result = classifier.test(inFile)
    print 'P@1:', result.precision
    print 'R@1:', result.recall
    print 'Number of examples:', result.nexamples


# # 获取测试数据
def getTestSet(inFile):
    # 训练集
    train_set = []
    # 情感标签集
    docid_set = []

    # 读入训练数据
    f = open(inFile)
    lines = f.readlines()
    for line in lines:
        article = line.replace('\n', '').split(" ")

        # 文章id
        docid_set.append(article[0])

        # 内容
        content = article[1:]
        train_set.append(content)

    f.close()
    return (train_set, docid_set)


# 把分词以后的结果写入文件
def writeFile(outputfile, newline):
    fw = open(outputfile, 'ab')
    # fw.write(newline.encode("utf-8"))
    fw.write(newline)
    fw.close()


def pred(inFile, model, outputfile):
    # 读入测试文件
    testData, docid = getTestSet(inFile)

    # for t in testData:
    #     for i in t:
    #         # print i
    #         print ""

    # load训练好的模型
    classifier = fasttext.load_model(model, label_prefix='__label__')
    for idx, text in enumerate(testData):
        tText = "".join([t.decode('utf-8') +' ' for t in text])
        print "######################"
        print "idx is:", idx
        print "tText is:", tText

        # label = classifier.predict(tText)

        # Or with probability
        label = classifier.predict_proba(tText,k=1)

        print "label is:",label
        print "label len",len(label)
        print "id is:",docid[idx]
        # print "%s:%s" % (docid[idx],label[idx][0])
        print "label[idx]",label[idx][0]

        newline = docid[idx] + " " + label[idx][0][0] + "\n"
        writeFile(outputfile, newline)

def loadMdl(mdl):
    model = fasttext.load_model(mdl, encoding='utf-8')

def main():
    # 定义文件路径
    dataPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360/data/"
    mdlPath = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360/model/"

    # 定义训练文件名
    inFile = dataPath + "train/train_output_title.tsv"

    # test 文件
    testFile = dataPath + "test/test_output_title.tsv"

    # test 文件
    outFile = dataPath + "test/pred_output_title.tsv"

    # 定义vec模型
    model_skipgram = mdlPath + "vec_skipgram"
    # 定义vec模型
    model_CBOW = mdlPath + "vec_CBOW"

    # 定义clf模型
    model_clf_sg = mdlPath + "clf_model_sg"

    # 定义clf模型
    model_clf_CBOW = mdlPath + "clf_model_CBOW"

    # 训练vector
    trainSkipgram(inFile,model_skipgram)

    # 训练vector
    #     trainCBOW(inFile,model_CBOW)

    vec_skipgram = model_skipgram + ".vec"
    vec_CBOW = model_CBOW + ".vec"

    # 分类 使用skipgram
    # trainCLF(inFile,model_clf_sg,vec_skipgram)

    #     trainCLF(inFile,model_clf_CBOW,vec_CBOW)

    # 定义clf模型
    bin_clf_sg = model_clf_sg + ".bin"

    # 定义clf模型
    bin_clf_CBOW = model_clf_CBOW + ".bin"

    # 测试
    # pred(testFile, bin_clf_sg, outFile)

if __name__ == '__main__':
    main()