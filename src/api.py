
from keras_yolo3.yolo import YOLO
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from logos import detect_logo, match_logo
from similarity import load_brands_compute_cutoffs
from utils import load_extractor_model, load_features, model_flavor_from_name, parse_input
import utils

from flask import Flask, request
from flask_restful import Resource, Api
# prepare Flask
app = Flask(__name__)
api = Api(app)

global graph
graph = tf.get_default_graph()

sim_threshold = 0.95
output_txt = 'out.txt'
filename = 'inception_logo_features_200_trunc2.hdf5'
yolo = YOLO(**{"model_path": 'keras_yolo3/yolo_weights_logos.h5',
                "anchors_path": 'keras_yolo3/model_data/yolo_anchors.txt',
                "classes_path": 'data_classes.txt',
                "score" : 0.05,
                "gpu_num" : 1,
                "model_image_size" : (416, 416),
                }
               )
save_img_logo, save_img_match = True, True
test_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data/test')

# get Inception/VGG16 model and flavor from filename
model_name, flavor = model_flavor_from_name(filename)
## load pre-processed features database
features, brand_map, input_shape = load_features(filename)

## load inception model
model, preprocess_input, input_shape = load_extractor_model(model_name, flavor)
my_preprocess = lambda x: preprocess_input(utils.pad_image(x, input_shape).astype(np.float32))

## load sample images of logos to test against
input_paths = ['test_batman.jpg', 'test_robin.png', 'test_lexus.png', 'test_champions.jpg',
               'test_duff.jpg', 'test_underarmour.jpg', 'test_golden_state.jpg']
input_labels = [ s.split('test_')[-1].split('.')[0] for s in input_paths]
input_paths = [os.path.join(test_dir, 'test_brands/', p) for p in input_paths]

# compute cosine similarity between input brand images and all LogosInTheWild logos
( img_input, feat_input, sim_cutoff, (bins, cdf_list)
) = load_brands_compute_cutoffs(input_paths, (model, my_preprocess), features, sim_threshold, timing=True)


class Detect(Resource):
    def post(self):
        file = request.files['image']
        img = Image.open(file)

        ## find candidate logos in image
        with graph.as_default():
            prediction, image = detect_logo(yolo, img)

            ## match candidate logos to input
            outtxt = match_logo(image, prediction, (model, my_preprocess), (feat_input, sim_cutoff, bins, cdf_list, input_labels))

        return outtxt


api.add_resource(Detect, '/detect')
if __name__ == '__main__':
    app.run()