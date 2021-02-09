
import tensorflow as tf
from tensorflow import keras

save_path = "D:\\new_ver_train_data"

# keyword
keyword_h5 = "D:\\new_ver_train_data\\keyword_model_parameter.h5"
keyword_tflite = "D:\\new_ver_train_data\\keyword_model_parameter.tflite"
keyword_pb = "D:\\new_ver_train_data\\keyword_model_parameter.pb"

# command
command_h5 = "D:\\new_ver_train_data\\command_model_parameter.h5"
command_tflite = "D:\\new_ver_train_data\\command_model_parameter.tflite"
command_pb = "D:\\new_ver_train_data\\command_model_parameter.pb"

# call
call_h5 = "D:\\new_ver_train_data\\call_model_parameter.h5"
call_tflite = "D:\\new_ver_train_data\\call_model_parameter.tflite"
call_pb = "D:\\new_ver_train_data\\call_model_parameter.pb"

# camera
camera_h5 = "D:\\new_ver_train_data\\camera_model_parameter.h5"
camera_tflite = "D:\\new_ver_train_data\\camera_model_parameter.tflite"
camera_pb = "D:\\new_ver_train_data\\camera_model_parameter.pb"

# picture
picture_h5 = "D:\\new_ver_train_data\\picture_model_parameter.h5"
picture_tflite = "D:\\new_ver_train_data\\picture_model_parameter.tflite"
picture_pb = "D:\\new_ver_train_data\\picture_model_parameter.pb"

# record
record_h5 = "D:\\new_ver_train_data\\record_model_parameter.h5"
record_tflite = "D:\\new_ver_train_data\\record_model_parameter.tflite"
record_pb = "D:\\new_ver_train_data\\record_model_parameter.pb"

# stop
stop_h5 = "D:\\new_ver_train_data\\stop_model_parameter.h5"
stop_tflite = "D:\\new_ver_train_data\\stop_model_parameter.tflite"
stop_pb = "D:\\new_ver_train_data\\stop_model_parameter.pb"

#end
end_h5 = "D:\\new_ver_train_data\\end_model_parameter.h5"
end_tflite = "D:\\new_ver_train_data\\end_model_parameter.tflite"
end_pb = "D:\\new_ver_train_data\\end_model_parameter.pb"

h5_list = [keyword_h5, command_h5, call_h5, camera_h5, picture_h5, record_h5, stop_h5, end_h5]
tflite_list = [keyword_tflite, command_tflite, call_tflite, camera_tflite, picture_tflite, record_tflite, stop_tflite, end_tflite]


## main
def main():

    for h5_one, tflite_one in zip(h5_list, tflite_list):

        model = keras.models.load_model(h5_one, compile=False)

        export_path = save_path
        model.save(export_path, save_format="tf")

        saved_model_dir = save_path
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        open(tflite_one, 'wb').write(tflite_model)


    return











##
if __name__ == '__main__':
    main()




## endl
