'''
import os
for root, dirs, files in os.walk('platforms'):
     for file in files:
        print file
        #with open(os.path.join(root, file), "r") as auto:
        with open(os.path.join(root, file),'r') as f:
            for line in f:
                words = [word for word in line.split()]
                print len(words)

'''
import os,sys
subdir  =  os.listdir('platforms')
#print subdir


labels = []
document_final = []
mydict =  []
for platform in subdir:
    #print 'in plat',platform
    path = 'platforms/' + str(platform)
    label  = subdir.index(platform)
    cnt = 0
    for root,dirs,files in os.walk(path):
        for file in files:
            cnt += 1
    dicti = {}
    dicti[platform]  = cnt
    mydict.append(dicti)

print 'MYDICT IS ',mydict
print 'len is ',len(mydict)

for dict in mydict:
    #print dict
    for key,val in dict.iteritems() :
        label = key
        if val > 720 and val < 14000 :

            path = 'platforms/' + str(key)

            for root,dirs,files in os.walk(path):
                for file in files:
                    with open(os.path.join(root,file),'r') as f:
                        doc = ''
                        for line in f:
                            doc = doc + str(line)
                    document_final.append(doc)
                    labels.append(label)



print 'lens is ', len(document_final),  len(labels)

labels2 = set(labels)
print 'uniques palts are ',labels2

import pandas as pd
df = pd.DataFrame()
df['label'] = labels
df['docs']  = document_final
df.to_csv('req_file2.csv')
