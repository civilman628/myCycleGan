from flask import Flask, Blueprint, abort
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin
from config import get_config
from model import cyclegan
from StringIO import *



import utils
import base64
import Image
import numpy as np
import tensorflow as tf
tf.set_random_seed(19)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


app = Flask(__name__)
CORS(app)  #

args = get_config()

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True

sess = tf.Session(config=tfconfig)
model = cyclegan(sess, args)
init_op = tf.global_variables_initializer()
sess.run(init_op)
model.load(args.checkpoint_dir)

out_var, in_var = (model.testB, model.test_A) if args.which_direction == 'AtoB' else (
    model.testA, model.test_B)


sample_image = [utils.load_test_data('/home/scopeserver/RaidDisk/DeepLearning/mwang/CycleGAN-tensorflow/datasets/dress/testB/164081901.jpg', args.fine_size)]
sample_image = np.array(sample_image).astype(np.float32)

newstyle_img = sess.run(out_var, feed_dict={in_var: sample_image})
newstyle_img = utils.inverse_transform(newstyle_img)
newstyle_img = utils.merge(newstyle_img,[1, 1])
newstyle_img = np.array(newstyle_img*255.0,dtype=np.int8)

image_buffer = StringIO()

newstyle_img = Image.fromarray(newstyle_img,'RGB')
newstyle_img.save(image_buffer,'png')
print base64.encodestring(image_buffer.getvalue())

#newstyle_img.save('temp.png')#



@app.route('/')
def index():
    return "This is fashion detection"


@app.route('/api/detect', methods=['POST'])
@cross_origin()
def create_task():

    if not request.json or not 'url' in request.json:
        abort(400)
    try:
        print 'test'
        result = myobj.detect_obj.demo(request.json['url'])
        return_data = {}
        return_data['result'] = 'OK'
        return_data['data'] = result
        return jsonify(return_data), 201
    except Exception as e:
        return_data = {}
        return_data['result'] = 'ERROR'
        return_data['json'] = str(request.json)
        return_data['errorMessage'] = str(e)
        return jsonify(return_data), 201


@app.route('/api/base64_style_transfer', methods=['POST'])
@cross_origin()
def create_base64_style_transfer_task():
    return_data = {}
    if not request.json or not 'base64' in request.json:
        abort(400)
    try:
        print 'base64'
        old_style_image = utils.string2image(request.json['base64'])
        old_style_image = np.array(old_style_image / 127.5 - 1).astype(np.float32)
        old_style_image = np.expand_dims(old_style_image, axis=0)

        newstyle_img = sess.run(out_var, feed_dict={in_var: old_style_image})
        newstyle_img = utils.inverse_transform(newstyle_img)
        newstyle_img = utils.merge(newstyle_img, [1, 1])
        newstyle_img = np.array(newstyle_img * 255.0, dtype=np.int8)

        newstyle_img = Image.fromarray(newstyle_img, 'RGB')
        image_buffer = StringIO()
        newstyle_img.save(image_buffer,'png')
        result = base64.encodestring(image_buffer.getvalue())

        return_data['result'] = 'OK'
        return_data['data'] = "data:image/jpeg;base64," + result
        return jsonify(return_data), 201
    except Exception as e:
        print str(e)
        return_data['result'] = 'ERROR'
        return_data['json'] = str(request.json)
        return_data['errorMessage'] = str(e)
        return jsonify(return_data), 201


if __name__ == '__main__':
        # print "aa"
    app.run(port=3535, debug=True, use_reloader=False, host='0.0.0.0')
