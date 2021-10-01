from flask import Flask, render_template, request, url_for
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename 
import os
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow as tf
import random
import string
 
UPLOAD_FOLDER = r'C:\Users\namay\OneDrive\Desktop\SEM4\MPR\cyclegan\static\img' 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','jfif']) 

app  = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods = ['POST', 'GET']) 
def home() : 
    if request.method == 'POST' : 
        f = request.files['myfile']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        input = plt.imread('./static/img/' + filename) 
        input.resize((256,256,3))
        input = tf.convert_to_tensor(input)
        input = (tf.cast(input, tf.float32) / 127.5 ) - 1
        model = keras.models.load_model('./monet_generator.h5')
        monet = model.predict(tf.reshape(input, [1,256,256,3]))
        output_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 6)) + '.png'
        tf.keras.utils.save_img('./static/img/' + output_filename, monet[0], data_format = 'channels_last', scale = True)
        output_path = r'img/' + output_filename 
        input_path = r'img/' + filename
        print(input_path, output_path)
        return render_template('index.html', output_path = r'img/' + output_filename, input_path = r'img/' + filename)
    return render_template('index.html')


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0',port=8080)

# image = plt.imread('./images/ss.jpg')
# image.resize((256,256,3))
# image = tf.convert_to_tensor(image)
# image = (tf.cast(image, tf.float32) / 127.5 ) - 1
# model = keras.models.load_model('./monet_generator.h5')
# monet = model.predict(tf.reshape(image, [1,256,256,3]))
# tf.keras.utils.save_img('./images/monet.png', monet[0], data_format = 'channels_last', scale = True)

# name = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 6)) + '.jpg'
# print(name)