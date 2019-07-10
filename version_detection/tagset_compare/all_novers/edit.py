from os import listdir
from os.path import isfile, join
import yaml

onlyfiles = [f for f in listdir('.') if isfile(join('.',f))]

for f in onlyfiles: 
    print(f)
    if f[0] != 'e':
        with open(f, 'r') as stream:
            data = yaml.safe_load(stream)

        fl = data['label'][0]
        nlab = '' 

        if fl == 'a':
            nlab = 'arpwatch'
        elif fl == 'c':
            nlab = 'crda'
        elif fl == 'd':
            nlab = 'dcraw'
        elif fl == 'g':
            nlab = 'gitweb'
        elif fl == 'q':
            nlab = 'quota'
        else:
            nlab = 'vsftpd'

        data['label'] = nlab

        with open(f, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)


