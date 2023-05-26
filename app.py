import numpy as np
import gradio as gr
from keras.models import load_model

model = load_model('digit_detector.h5')

def detect_digit(img):
  # the image should be of the same dimension as the images used to train i.e 20*20
  pred_logit = model.predict(img.reshape(1, 400))
  pred = np.argmax(pred_logit)
  return pred

app = gr.Interface(fn=detect_digit, inputs='image', outputs='label',
                   title="Digit Detection",
                   description="Who needs a description")
app.launch()