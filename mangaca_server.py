import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # mute all the cuda device warnings
import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model
import cv2
from skimage.measure import block_reduce
from pathlib import Path
import http.server


# image encoding conversions
def ToGray(x):
    R = x[:, :, :, 0:1]
    G = x[:, :, :, 1:2]
    B = x[:, :, :, 2:3]
    return 0.30 * R + 0.59 * G + 0.11 * B


def RGB2YUV(x):
    R = x[:, :, :, 0:1]
    G = x[:, :, :, 1:2]
    B = x[:, :, :, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = 0.492 * (B - Y) + 128
    V = 0.877 * (R - Y) + 128
    return tf.concat([Y, U, V], axis=3)


def YUV2RGB(x):
    Y = x[:, :, :, 0:1]
    U = x[:, :, :, 1:2]
    V = x[:, :, :, 2:3]
    R = Y + 1.140 * (V - 128)
    G = Y - 0.394 * (U - 128) - 0.581 * (V - 128)
    B = Y + 2.032 * (U - 128)
    return tf.concat([R, G, B], axis=3)


def VGG2RGB(x):
    return (x + [103.939, 116.779, 123.68])[:, :, :, ::-1]


print('Begin Model Init')
# setup
session = keras.backend.get_session()

with tf.device('/gpu:0'):
    # network inputs
    ipa = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 1))
    ip1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None, None, 1))
    ip3 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
    ip4 = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None, None, 4))
    ip3x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

    print('Loading Apprentice Ini')
    # global hint generator
    apprentice_ini = load_model('Models/apprentice_ini.net')
    input = tf.concat([- 512 * tf.ones_like(ip4[:, :, :, 3:4]), 128 * tf.ones_like(ip4[:, :, :, 3:4]), 128 * tf.ones_like(ip4[:, :, :, 3:4])], axis=3)
    apprentice_ini_yuv = RGB2YUV(ip4[:, :, :, 0:3])
    apprentice_ini_alpha = tf.where(x=tf.zeros_like(ip4[:, :, :, 3:4]), y=tf.ones_like(ip4[:, :, :, 3:4]), condition=tf.less(ip4[:, :, :, 3:4], 128))
    apprentice_ini_hint = apprentice_ini_alpha * apprentice_ini_yuv + (1 - apprentice_ini_alpha) * input
    apprentice_ini_op = YUV2RGB(apprentice_ini(tf.concat([ip1, apprentice_ini_hint], axis=3)))

    print('Loading Apprentice')
    # global hint generator
    apprentice = load_model('Models/apprentice.net')
    apprentice_op = (1 - apprentice([1 - ip1 / 255.0, ip4, 1 - ip3 / 255.0])) * 255.0

    print('Loading Feature Extractor')
    # feature extraction net
    feature_extractor = load_model('Models/feature_extractor.net')
    features = feature_extractor(ip3 / 255.0)
    featuresx = feature_extractor(ip3x / 255.0)

    print('Loading Painter')
    # feed into head.net features extracted by reader.net at every block
    painter = load_model('Models/painter.net')
    feed = [1 - ip1 / 255.0, (ip4[:, :, :, 0:3] / 127.5 - 1) * ip4[:, :, :, 3:4] / 255.0]
    for _ in range(len(features)):
        item = keras.backend.mean(features[_], axis=[1, 2])
        itemx = keras.backend.mean(featuresx[_], axis=[1, 2])
        feed.append(item * ipa + itemx * (1 - ipa))
    nil0, nil1, head_temp = painter(feed)

    print('Loading Painter Alt')
    # alt for head.net, employs the same structure but has diff weights set
    painter_alt = load_model('Models/painter_alt.net')
    nil2, nil3, neck_temp = painter_alt(feed)
    feed[0] = tf.clip_by_value(1 - tf.compat.v1.image.resize_bicubic(ToGray(VGG2RGB(head_temp) / 255.0), tf.shape(ip1)[1:3]), 0.0, 1.0)
    nil4, nil5, head_temp = painter_alt(feed)
    painter_op = VGG2RGB(head_temp)
    painter_alt_op = VGG2RGB(neck_temp)

    ip3B = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

    print('Loading UpRes')
    # short image upscaling net
    up_res = load_model('Models/up_res.net')
    pads = 7
    up_res_op = up_res(tf.pad(ip3B / 255.0, [[0, 0], [pads, pads], [pads, pads], [0, 0]], 'REFLECT'))[:, pads * 2:-pads * 2, pads * 2:-pads * 2, :] * 255.0
    session.run(tf.global_variables_initializer())

print('Loading Weights')
# load model weights
up_res.load_weights('Models/up_res.net')
painter.load_weights('Models/painter.net')
painter_alt.load_weights('Models/painter_alt.net')
apprentice.load_weights('Models/apprentice.net')
feature_extractor.load_weights('Models/feature_extractor.net')
apprentice_ini.load_weights('Models/apprentice_ini.net')

print('Init Completed')
print('\nApprentice_ini model:')
apprentice_ini.summary()
# tf.keras.utils.plot_model(apprentice_ini, to_file='apprentice_ini.png', show_shapes=True)
print('\nApprentice model:')
apprentice.summary()
# tf.keras.utils.plot_model(apprentice, to_file='apprentice.png', show_shapes=True)
print('\nfeature_extractor model:')
feature_extractor.summary()
# tf.keras.utils.plot_model(feature_extractor, to_file='feature_extractor.png', show_shapes=True)
print('\npainter model:')
painter.summary()
# tf.keras.utils.plot_model(painter, to_file='painter.png', show_shapes=True)
print('\npainter_alt model:')
painter_alt.summary()
# tf.keras.utils.plot_model(painter_alt, to_file='painter_alt.png', show_shapes=True)
print('\nup_res model:')
up_res.summary()
# tf.keras.utils.plot_model(up_res, to_file='up_res.png', show_shapes=True)


def run_painter(sketch, global_hint, local_hint, global_hint_x, alpha):
    return session.run(painter_op, feed_dict={
        ip1: sketch[None, :, :, None], ip3: global_hint[None, :, :, :], ip4: local_hint[None, :, :, :], ip3x: global_hint_x[None, :, :, :], ipa: np.array([alpha])[None, :]
    })[0].clip(0, 255).astype(np.uint8)


def run_painter_alt(sketch, global_hint, local_hint, global_hint_x, alpha):
    return session.run(painter_alt_op, feed_dict={
        ip1: sketch[None, :, :, None], ip3: global_hint[None, :, :, :], ip4: local_hint[None, :, :, :], ip3x: global_hint_x[None, :, :, :], ipa: np.array([alpha])[None, :]
    })[0].clip(0, 255).astype(np.uint8)


def run_apprentice(sketch, latent, hint):
    return session.run(apprentice_op, feed_dict={
        ip1: sketch[None, :, :, None], ip3: latent[None, :, :, :], ip4: hint[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)


def run_up_res(x):
    return session.run(up_res_op, feed_dict={
        ip3B: x[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)


def run_apprentice_ini(sketch, local_hint):
    return session.run(apprentice_ini_op, feed_dict={
        ip1: sketch[None, :, :, None], ip4: local_hint[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)


# util functions

def k_resize(x, k):
    if x.shape[0] < x.shape[1]:
        s0 = k
        s1 = int(float(x.shape[1]) * float(k) / float(x.shape[0]))
        s1 = s1 - s1 % 64
        _s0 = 16 * s0
        _s1 = int(float(x.shape[1]) * float(_s0) / float(x.shape[0]))
        _s1 = (_s1 + 32) - (_s1 + 32) % 64
    else:
        s1 = k
        s0 = int(float(x.shape[0]) * float(k) / float(x.shape[1]))
        s0 = s0 - s0 % 64
        _s1 = 16 * s1
        _s0 = int(float(x.shape[0]) * float(_s1) / float(x.shape[1]))
        _s0 = (_s0 + 32) - (_s0 + 32) % 64
    new_min = min(_s1, _s0)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
    return y


def sk_resize(x, k):
    if x.shape[0] < x.shape[1]:
        s0 = k
        s1 = int(float(x.shape[1]) * (float(k) / float(x.shape[0])))
        s1 = s1 - s1 % 16
        _s0 = 4 * s0
        _s1 = int(float(x.shape[1]) * (float(_s0) / float(x.shape[0])))
        _s1 = (_s1 + 8) - (_s1 + 8) % 16
    else:
        s1 = k
        s0 = int(float(x.shape[0]) * (float(k) / float(x.shape[1])))
        s0 = s0 - s0 % 16
        _s1 = 4 * s1
        _s0 = int(float(x.shape[0]) * (float(_s1) / float(x.shape[1])))
        _s0 = (_s0 + 8) - (_s0 + 8) % 16
    new_min = min(_s1, _s0)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
    return y


def d_resize(x, d, fac=1.0):
    new_min = min(int(d[1] * fac), int(d[0] * fac))
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
    return y


def ini_hint(x):
    r = np.zeros(shape=(x.shape[0], x.shape[1], 4), dtype=np.float32)
    return r


def min_k_down(x, k):
    y = 255 - x.astype(np.float32)
    y = block_reduce(y, (k, k), np.max)
    y = 255 - y
    return y.clip(0, 255).astype(np.uint8)


def min_k_down_c(x, k):
    y = 255 - x.astype(np.float32)
    y = block_reduce(y, (k, k, 1), np.max)
    y = 255 - y
    return y.clip(0, 255).astype(np.uint8)


def mini_norm(x):
    y = x.astype(np.float32)
    y = 1 - y / 255.0
    y -= np.min(y)
    y /= np.max(y)
    return (255.0 - y * 80.0).astype(np.uint8)


def hard_norm(x):
    o = x.astype(np.float32)
    b = cv2.GaussianBlur(x, (3, 3), 0).astype(np.float32)
    y = (o - b + 255.0).clip(0, 255)
    y = 1 - y / 255.0
    y -= np.min(y)
    y /= np.max(y)
    y[y < np.mean(y)] = 0
    y[y > 0] = 1
    return (255.0 - y * 255.0).astype(np.uint8)


def clip_15(x, s=15.0):
    return ((x - s) / (255.0 - s - s)).clip(0, 1) * 255.0


def ini_hint(x):
    r = np.zeros(shape=(x.shape[0], x.shape[1], 4), dtype=float)
    return r


def apply_hint_points(hint, points, size):
    h = hint.shape[0]
    w = hint.shape[1]
    for point in points:
        x, y, r, g, b = point
        x = int(x * w)
        y = int(y * h)
        l_ = max(0, x - size)
        b_ = max(0, y - size)
        r_ = min(w, x + size + 1)
        t_ = min(h, y + size + 1)
        hint[b_:t_, l_:r_, 2] = r
        hint[b_:t_, l_:r_, 1] = g
        hint[b_:t_, l_:r_, 0] = b
        hint[b_:t_, l_:r_, 3] = 255.0
    return hint


def colorize_image(src_image, hint_points, use_global_hint=True, use_alt_painter=False):
    sketch = src_image
    sketch_1024 = k_resize(sketch, 64)
    sketch_256 = mini_norm(k_resize(min_k_down(sketch_1024, 2), 16))
    sketch_128 = hard_norm(sk_resize(min_k_down(sketch_1024, 4), 32))
    print('sketch prepared')

    hint_128 = apply_hint_points(ini_hint(sketch_128), hint_points, 1)
    hint_256 = ini_hint(sketch_256)
    hint_1024 = apply_hint_points(ini_hint(sketch_1024), hint_points, 2)

    g_hint = run_apprentice_ini(sketch_128, hint_128)
    g_hint = run_up_res(g_hint)
    g_hint = clip_15(g_hint)
    print('sketch preprocessed')

    g_hint = run_apprentice(sketch=sketch_256, latent=d_resize(g_hint, sketch_256.shape), hint=hint_256) if use_global_hint else g_hint
    g_hint = run_up_res(g_hint)
    if use_global_hint:
        print('global hint prepared')

    paint = run_painter_alt if use_alt_painter else run_painter

    result = paint(
        sketch=sketch_1024,
        global_hint=k_resize(g_hint, 14),
        local_hint=hint_1024,
        global_hint_x=k_resize(g_hint, 14),
        alpha=1
    )
    result = run_up_res(result)
    print('colorization completed')

    return result


import cgi
import json
import base64
import random
import requests

# flask
from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/api", methods=["POST"])
def main():
    if request.json is None:
        return 'Invalid Request: "Did not receive valid JSON data."'

    post_data = request.json

    if 'image' not in post_data:
        return 'Invalid Request: "Malformed JSON data."'

    # decode image bytes n write to tmp file
    img_data = base64.b64decode(post_data['image'].split(',')[1])
    ext = 'png'

    tmp_path = 'tmp/mangaca_%d.%s' % (random.randint(0, 10000000), ext)
    with open(tmp_path, 'wb') as tmp_file:
        tmp_file.write(img_data)
        tmp_file.close()

    # load image, rm tmp file
    src_image = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
    os.remove(tmp_path)

    # colorize and encode as png
    dst_image = colorize_image(src_image, post_data['hint_points'])
    png_data = cv2.imencode('.png', dst_image)[1]
    base64_encoded = 'data:image/png;base64,' + base64.b64encode(png_data).decode('utf8')

    print('done')
    try:
        return base64_encoded
    except:
        return 'Other errors'


app.run()
