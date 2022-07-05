# file processing modules
import os


class FileProcessing():

    # global variables
    target_files_list = list()
    
    def get_target_files_list(self):
        return self.target_files_list

    def set_target_files_list(self, set_target):
        self.target_files_list = set_target

    # constructor
    def __init__(self):
        return

    # generate files list for target path
    def gen_target_files_list(self, files_path, file_ext):

        result_list = list()

        print('started finding target files...')
        print('target path : {path}, target ext : {ext}'.format(path=files_path, ext=file_ext))

        for (path, dir, files) in os.walk(files_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == file_ext:
                    file_name = path + '/' + filename
                    result_list.append(file_name)

        print('return result...')
        self.set_target_files_list(result_list)




## test 
print('hello, world~!!')

test_class = FileProcessing()

target_path = "/home/jy/.devjy/girl_from_ipanema/"
target_ext = '.py'

test_class.gen_target_files_list(target_path, target_ext)
print(test_class.get_target_files_list())




