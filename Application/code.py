from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import os
import io
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
  
try:
    import shutil
    shutil.rmtree('uploaded / image')
    print()
except:
    pass
  
model=  load_model("effnet.h5")
app = Flask(__name__)
  
app.config['UPLOAD_FOLDER'] = 'uploaded\\image'
  
@app.route('/')
def upload_f():
    return render_template('upload.html')
  
def finds(fpath):
    imdir = 'uploaded'
    img = Image.open(fpath)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage,(150,150))
    img = img.reshape(1,150,150,3)
    p = model.predict(img)
    p = np.argmax(p,axis=1)[0]
    if p==0:
        tumour='Glioma Tumor'
        print('Glioma Tumor')
    elif p==1:
        tumour='No Tumor'
        print('No Tumor')
    elif p==2:
        tumour='Meningioma Tumor'
        print('Meningioma Tumor')
    else:
        tumour='Pituitary Tumor'
        print('Pituitary Tumor')
    return tumour
  
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        fpath=os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(fpath)
        val = finds(fpath)
        return render_template('pred.html', ss = val)
  
if __name__ == '__main__':
    app.run()