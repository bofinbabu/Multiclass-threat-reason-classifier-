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
print subdir


labels = []
document_final = []

for platform in subdir:
    print 'in plat',platform
    path = 'platforms/' + str(platform)
    label  = subdir.index(platform)
    for root,dirs,files in os.walk(path):

        for file in files:
            with open(os.path.join(root,file),'r') as f:
                doc = ''
                for line in f:
                    doc = doc + str(line)
                document_final.append(doc)
                labels.append(label)

import pandas as pd

#labels = [1,2,3]
#documents = ['djs','fhdj','fiu']

df = pd.DataFrame()
df['label'] = labels
df['docs']  = document_final
print df[:5]
df.to_csv('req_file.csv')
