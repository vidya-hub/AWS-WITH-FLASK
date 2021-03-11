import os
from flask import Flask, send_file, request, render_template
import boto3
import numpy as np
import io
import base64
from cv2 import cv2
from PIL import Image
filename = f"{str(os.getcwd())+'/'+'uploads'}"
print(filename)
app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = filename


def detect_faces_from_localfile(photo):
    client = boto3.client('rekognition')
    with open(photo, 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()})
    return response


def imageConverter(imageArray):
    img = Image.fromarray(imageArray.astype('uint8'))
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    encoded_img_data = base64.b64encode(file_object.getvalue())
    imageio = encoded_img_data.decode('utf-8')
    return imageio


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/save', methods=['POST'])
def save():
    faceslist = []
    facesroi = []
    
    image = request.files["file"]
    fileextension=str(image.filename.split(".")[1]).lower()
    if(fileextension=="jpg" or fileextension=="jpeg" or fileextension=="png"):
        img_file_storage = os.path.join(
            app.config['IMAGE_UPLOADS'], image.filename)
        image.save(img_file_storage)
        res = detect_faces_from_localfile(str(img_file_storage))
        image = Image.open(open(img_file_storage, 'rb'))
        iwidth, iheight = image.size
        faces = res["FaceDetails"]
        for face in faces:
            faceslist.append(face)
        image = cv2.imread(img_file_storage)
        os.remove(img_file_storage)
        print(len(faceslist))
        if len(faceslist) != 0:
            for face in faceslist:
                for lan in face["Landmarks"]:
                    cv2.circle(image, (round(lan["X"]*iwidth),
                                    round(lan["Y"]*iheight)), 5, (0, 0, 255), -1)
                left = iwidth * face["BoundingBox"]['Left']
                top = iheight * face["BoundingBox"]['Top']
                width = iwidth * face["BoundingBox"]['Width']
                height = iheight * face["BoundingBox"]['Height']
                ROI = image[round(top):round(top)+round(height),
                            round(left):round(left)+(round(width))]
                rgbimage = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
                conimg = imageConverter(rgbimage)
                facesroi.append(conimg)
                cv2.rectangle(image, (round(left), round(top)),
                            (round(left+width), round(top+height)), (0, 255, 0), 2)
            imageoutput = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            arr = np.array(imageoutput)
            realimageio = imageConverter(arr)
            return render_template("output.html", result="AWS RESULT FACES", img_data=realimageio, facelist=facesroi)

        else:
            rgbimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            conimg = imageConverter(rgbimage)
            return render_template("output.html", result="No Faces Found", img_data=conimg, facelist=[])
    else:
        return "Upload Image files only"

if __name__ == '__main__':
    app.run(debug=True)
