import subprocess
import os
directory = input("Enter directory name: ")
full_dir_path = os.path.abspath(directory)

for root,dirs,files in os.walk(full_dir_path):
    for file in files: 
        full_file_path = os.path.join(root, file)
        print(full_file_path) #Checking to see the path
        subprocess.call(['wc','-l',full_file_path])
