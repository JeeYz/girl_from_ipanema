import os
'''
class : utility functions
method0 : find files
            -> arg0 : target_path
'''

class utility_functions():

    def __init__(self):
        return


    def find_files(self, **kwargs):

        files_list = list()

        if 'target_path' in kwargs.keys():
            target_path = kwargs['target_path']
        if 'target_file_format' in kwargs.keys():
            target_file_format = kwargs['target_file_format']

        for (path, dir, files) in os.walk(target_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == target_file_format:
                    print("%s/%s" % (path, filename))
                    self.files_list.append(path+'\\'+filename)
    
        return files_list






