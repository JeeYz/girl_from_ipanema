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

        if 'target_path' in kwargs.keys():
            target_path = kwargs['target_path']

        for (path, dir, files) in os.walk(target_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.py':
                    print("%s/%s" % (path, filename))
    
        return
