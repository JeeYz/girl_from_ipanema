import os
'''
class : utility functions
method0 : find files
            -> arg0 : target_path
'''

class utility_functions():

    def __init__(self):
        self.files_list = list()
        return


    def find_files(self, **kwargs):

        if 'target_path' in kwargs.keys():
            target_path = kwargs['target_path']
        if 'target_ext' in kwargs.keys():
            target_ext = kwargs['target_ext']

        for (path, dir, files) in os.walk(target_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == target_ext:
                    print("%s/%s" % (path, filename))
                    self.files_list.append(path+'\\'+filename)
    
        return
