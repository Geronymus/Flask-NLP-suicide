from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

app = Flask(__name__)


model_name_emotions = 'D:\WorkSpacex\Modelos_software\Modelos_software_back_up_two\model_output_emotions_90'
tokenizer_emotions = BertTokenizer.from_pretrained(model_name_emotions)
model_emotions = BertForSequenceClassification.from_pretrained(model_name_emotions)

model_name_suicidio = 'D:\WorkSpacex\Modelos_software\Modelos_software_back_up_two\model_output_suicidio_3OK'
tokenizer_suicidio = BertTokenizer.from_pretrained(model_name_suicidio)
model_suicidio = BertForSequenceClassification.from_pretrained(model_name_suicidio)

def make_prediction_suicide(text):
    inputs = tokenizer_suicidio(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model_suicidio(**inputs)
        predictions = softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions)
    return predicted_class.item()

def make_prediction_emotions(text):
    inputs = tokenizer_emotions(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model_emotions(**inputs)
        predictions = softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions)
    return predicted_class.item()


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result_suicide = None
    prediction_result_emotion = None
    if request.method == 'POST':
        text = request.form['user_text']
        prediction_suicide = make_prediction_suicide(text)
        prediction_emotion = make_prediction_emotions(text)
        
        if prediction_suicide == 0:
            prediction_result_suicide = "Comentario no suicida"
        elif prediction_suicide == 1: 
            prediction_result_suicide = "Comentario suicida"
        if prediction_emotion == 0:
            prediction_result_emotion = "Comentario triste"
        elif prediction_emotion == 1: 
            prediction_result_emotion = "Comentario furioso"
        elif prediction_emotion == 2: 
            prediction_result_emotion = "Comentario con miedo"

    return render_template('index.html', prediction_suicide=prediction_result_suicide,prediction_emotion=prediction_result_emotion)

if __name__ == '__main__':
    app.run(debug=True)
