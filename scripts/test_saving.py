import os
import shutil
import tensorflow as tf
import sys
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
import sys
import ssd_keras.bounding_box_utils.bounding_box_utils
from ssd_keras.models.keras_ssd512 import ssd_512
from ssd_keras.keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_keras.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_keras.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from ssd_keras.keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from ssd_keras.keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_keras.ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from ssd_keras.data_generator.object_detection_2d_data_generator import DataGenerator
from ssd_keras.data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from ssd_keras.data_generator.object_detection_2d_geometric_ops import Resize
from ssd_keras.data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


img_height = 512
img_width = 512

def save_model(model: tf.keras.Model, path: str) -> None:
    """
    Save the model to a specified path. The method here is recommende by the AWS SageMaker Neo
    team in order to output a frozen graph model that is compatible with the AWS Panorama device.

    :param model: a keras model
    :param path: path to save the model artifact
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    # Save as SavedModel
    tmp_dir = 'tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    model.save(tmp_dir, save_format='tf', include_optimizer=False)

    # Reload and save as frozen graph
    output_node_names = [output.name.split(":")[0] for output in model.outputs]

    with tf.Session() as sess:
        loaded = tf.saved_model.load(sess, export_dir=tmp_dir, tags=["serve"])
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            output_node_names)
        filepath = os.path.join(path, 'frozen_graph.pb')
        tf.io.write_graph(graph_or_graph_def=frozen_graph,
                          logdir=".",
                          name=filepath,
                          as_text=False)
    shutil.rmtree(tmp_dir)

K.clear_session() # Clear previous models from memory.

model = ssd_512(image_size=(img_height, img_width, 3),
                n_classes=4,
                mode='training',
                l2_regularization=0.0005,
                scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
               two_boxes_for_ar1=True,
               steps=[8, 16, 32, 64, 128, 256, 512],
               offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               clip_boxes=False,
               variances=[0.1, 0.1, 0.2, 0.2],
               normalize_coords=True,
               subtract_mean=[123, 117, 104],
               swap_channels=[2, 1, 0],
               confidence_thresh=0.5,
               iou_threshold=0.45,
               top_k=200,
               nms_max_output_size=400)

weights_path = 'weights/primary_detector/ssd_keras/VGG_coco_SSD_512x512_iter_360000_4_classes.h5'

model.load_weights(weights_path, by_name=True)

model.summary()

save_model(model, 'saved')