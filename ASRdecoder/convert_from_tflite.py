
import tensorflow as tf
from tensorflow import keras


## main
def main():


    model = keras.models.load_model("D:\\resnet_model_all.h5", compile=False)

    export_path = 'D:\\h5_pb_file'
    model.save(export_path, save_format="tf")


    saved_model_dir = 'D:\\h5_pb_file'
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open('D:\\h5_pb_file\\converted_model_resnet_all_ver_3.tflite', 'wb').write(tflite_model)


    return











##
if __name__ == '__main__':
    main()




## endl
