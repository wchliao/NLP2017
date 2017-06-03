import numpy as np
import codecs
import pickle

f = codecs.open('/tmp2/eee/K/zh.tsv'.format('zh'), 'r', 'utf-8')
w2v_dict = {}
key = ""
values = []
for lines in f:
    terms = lines.split('\t')
    if len(terms) == 3:
        if key != "":
            w2v_dict[key] = values
        key = terms[1]
    else:
        values += [float(v.replace('\n','').replace('[','').replace(']','')) for v in terms[0].split(' ') if v.replace('\n','').replace('[','').replace(']','') != '']

#pickle.dump( w2v_dict, open( "/tmp2/eee/K/w2v_dict.p", "wb" ) )
print('Done')
