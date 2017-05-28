"""
Preprocessing module for National Taiwan University Computer Science dept.
Natural Language Processing course Second Homework.

Target: connect to Paper: A Recurrent Neural Model with Attention for
the Recognition of Chinese Implicit Discourse Relations Samuel (ACL '17)
"""

def read(filename, train):
    with open(filename, 'r') as file:
        header = file.readline()[:-1].split()
        if train and len(header) != 4:
            print ('Training file with header: ', header, '... please check!')
            return None
        if not train and len(header) != 3:
            print ('Testing file with header: ', header, '... please check!')
            return None

        for line in file.readlines():
            segs = line.readline()[:-1].split()

            

    



