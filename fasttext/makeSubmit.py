f1 = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360/data/test/test_output_title.tsv"
f2 = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360/data/test/predict.txt"

f = open(f1)
docid = []
for line in f:
    docid.append(line.strip().split(' ')[0])

g = open(f2)
pred = []
for line in g:
    pred.append(line.strip().split('_')[4])

out = zip(docid,pred)
print len(out)

submit = "/home/hadoop/DataSencise/bdci2017/BDCI2017-360/data/test/submit.csv"
fw = open(submit, 'ab')
for (id,pred) in out:
    newline = id + "," + pred + "\n"
    fw.write(newline.encode('utf8'))
fw.close()