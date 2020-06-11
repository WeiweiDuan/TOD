import os

def create_folder(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")
        
def remove_files(path):
    for root, directory, files in os.walk(path):
        for fname in files:
            os.remove(os.path.join(root, fname))
    return 0
