import os
from flask import Flask,send_file, request,render_template
import boto3
import numpy as np
import io
import base64
from cv2 import cv2
from PIL import Image
filename=f"{str(os.getcwd())+'/'+'uploads'}"
print(filename)
app=Flask(__name__)
app.config['IMAGE_UPLOADS']=filename

def detect_faces_from_localfile(photo):
    client = boto3.client('rekognition')
    with open(photo, 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()})
    return response

# def imageConverter(imageArray):



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save():
    faceslist = []
    facial = []
    image = request.files["file"]
    img_file_storage=os.path.join(app.config['IMAGE_UPLOADS'],image.filename)
    image.save(img_file_storage)
    res=detect_faces_from_localfile(str(img_file_storage))
    image = Image.open(open(img_file_storage, 'rb'))
    iwidth, iheight = image.size
    faces = res["FaceDetails"]
    for face in faces:
        faceslist.append(face)
    image = cv2.imread(img_file_storage)
    os.remove(img_file_storage)
    for face in faceslist:
        for lan in face["Landmarks"]:
            cv2.circle(image,(round(lan["X"]*iwidth),
                round(lan["Y"]*iheight)),5,(0,0,255),-1)
        left = iwidth * face["BoundingBox"]['Left']
        top = iheight * face["BoundingBox"]['Top']
        width = iwidth * face["BoundingBox"]['Width']
        height = iheight * face["BoundingBox"]['Height']
        ROI = image[round(top):round(top)+round(height),
                    round(left):round(left)+(round(width))]

        cv2.rectangle(image, (round(left), round(top)),
                        (round(left+width), round(top+height)), (0, 255, 0), 2)
    
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    arr = np.array(image)

    img = Image.fromarray(arr.astype('uint8'))
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    encoded_img_data = base64.b64encode(file_object.getvalue())
    
    return render_template("output.html", img_data=encoded_img_data.decode('utf-8'))



if __name__ == '__main__':
    app.run(debug=True)