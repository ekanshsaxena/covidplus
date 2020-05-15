import os
from flask import Flask, request, render_template, send_from_directory, redirect,url_for
import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
#from google.colab.patches import cv2_imshow
lst = ['COVID-19 prone','Normal Lungs']
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5',compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    folder='images'
    ex=folder+'/'+filename
    image = Image.open(ex)
    img=cv2.imread(ex)
    img = cv2.resize(img, (0, 0), fx = 0.2, fy = 0.2)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    if len(image_array.shape) == 2: # ----------------Change here
        image_array.resize(224, 224, 1)

    # display the resized image
    #image.show()

    #cv2_imshow(img)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array 

    # run the inference
    prediction = model.predict(data)
    #print(type(prediction))
    prediction = list(prediction)

    print(prediction)
    pred_prone=prediction[0][0]*100
    pred_normal=(100-pred_prone)*100
    return render_template("complete_display_image.html",image_name=filename,pred_prone=pred_prone,pred_normal=pred_normal)
    #print(type(prediction))

    #if (prediction[0][0]>prediction[0][1]):
      #pred=prediction[0][0]*100
      #return render_template("complete_display_image.html", image_name=filename, prediction=pred)

    #else:
      #pred=prediction
      #return render_template("complete_display_image.html", image_name=filename, prediction=lst[1])


@app.route("/ct_upload", methods=["POST"])
def ct_upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    # Load the model
    model = tensorflow.keras.models.load_model('ct_model.h5',compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    folder='images'
    ex=folder+'/'+filename
    image = Image.open(ex)
    img=cv2.imread(ex)
    img = cv2.resize(img, (0, 0), fx = 0.2, fy = 0.2)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    if len(image_array.shape) == 2: # ----------------Change here
        image_array.resize(224, 224, 1)

    # display the resized image
    #image.show()

    #cv2_imshow(img)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array 

    # run the inference
    prediction = model.predict(data)
    #print(type(prediction))
    prediction = list(prediction)

    print(prediction)
    #print(type(prediction))

    if (prediction[0][0]>prediction[0][1]):
      return render_template("complete_display_image.html", image_name=filename, prediction=lst[0])

    else:
      return render_template("complete_display_image.html", image_name=filename, prediction=lst[1])

    
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/go back')
def back():
    return redirect("http://covidplus.live", code=302)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
