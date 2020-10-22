from flask import Flask, request, redirect, url_for, render_template
# from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
# import tensorflow as tf
from tensorflow.keras.models import model_from_json
import requests
from keras.utils.data_utils import get_file
#import keras
from PIL import Image
import os, time
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from torch import argmax, tensor
print('imported libraries!')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
print('Bert Model loaded!')
features = ['female', 'male', 'abdomen', 'acral', 'back', 'chest', 'ear','face', 'foot', 'genital', 'hand', 'lower extremity', 'neck','scalp', 'trunk', 'unknown', 'upper extremity', 'confocal','consensus', 'follow up', 'histo', 'age']

app = Flask(__name__,static_folder=os.path.abspath('static/'))
app.secret_key = "secret"

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "GET":
        print("IN get")
        return render_template("index.html")

    if request.method == "POST":
        print("Imported")
        #Random Forest

        #CNN model
        json_file = open('model_shape.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        print("cnn model")
        loaded_model.load_weights("model_weights.h5")
        print("Loaded Weights")

        # json_file2 = open('completemodel1.json', 'r')
        # loaded_model_json2 = json_file2.read()
        # json_file2.close()
        # complete_model = model_from_json(loaded_model_json2)
        # print("cnn model")
        # complete_model.load_weights("completeweights.h5")

        name = request.form["name"]
        age = int(request.form['age'])
        gender = request.form['gender']
        typ = request.form['type']
        localization = request.form['localization']
        f = request.files['myfile']
        image_path = 'static/images/img-'+str(int(time.time()))+'-'+str(f.filename)
        f.save(image_path)
        
        print(image_path)

        parameters = [0]*len(features)

        try:
            parameters[features.index(gender.lower())] = 1
        except:
            pass

        print(name)
        parameters[features.index('age')] = age
        parameters[features.index(typ.lower())] = 1
        parameters[features.index(localization.lower())] = 1

        print('loaded_model')
        img = Image.open(image_path).resize((71,71))
        img = np.asarray(img)
        img = img.reshape(-1,71,71,3)
        img = img/255

        print("IMAGE")
        print(img)

        # ## images prob
        # image_proba = complete_model.predict(img)
        # image_proba = image_proba[0]    
        # print("image proba")
        # print(image_proba)

        print("parameters:::")
        print(parameters)

        pred = loaded_model.predict(img)
        print("pred:::::")
        print(pred)
        print(len(pred))
        print(pred[0])

        cols = ['female', 'male', 'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'unknown', 'upper extremity', 'confocal', 'consensus', 'follow_up', 'histo', 'age', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51']

        parameters.extend(list(pred[0]))
        print(pred)
        print(parameters)
        del loaded_model

        temp_dict = dict(zip(cols,parameters),index=[0])
        temp_df = pd.DataFrame(temp_dict)
        temp_df = temp_df.drop(['index'],axis=1)
        print(temp_df.head(n=1))
        diseases = ['Basal cell carcinoma','Actinic keratoses','Benign keratosis-like lesions','Dermatofibroma','Melanocytic nevi','Melanoma','Vascular lesions']
        
        xgb = pickle.load(open("xgboost", 'rb'))
        print("xgboost")
        preds = xgb.predict_proba(temp_df)

        print("preds",preds)
        
        preds = list(preds[0])
        preds = [int(x*100) for x in preds]

        print(preds)

        data_df = pd.DataFrame(list(zip(diseases,preds)),columns=['disease','preds'])
        data_df.sort_values(by=['preds'],ascending = False , inplace = True)
        data = {
            'disease' : data_df['disease'].tolist()[:3],
            'prob' : data_df['preds'].tolist()[:3]
        }


        print(data)
        return render_template("result.html",len=3,disease=data["disease"],prob = data["prob"],imagepath=image_path)

@app.route('/getanswer', methods=['POST'])
def get_answer():
    data = request.get_json()

    question = data['question']
    disease = data['disease']

    filename = disease.split(" ")[0].lower()
    filename = filename + ".txt"
    print("In get answer")
    # filename = "demo.txt"
    with open("static/passage/"+filename , 'r') as file:
        passage = file.read().replace('\n', '')
    
    answer = answer_question(question, passage, model, tokenizer)

    return answer


@app.route('/near-by-doctors', methods=['GET'])
def get_nearby():
    print('here')
    latitude = request.args.get('lat')
    longitude = request.args.get('long')

    print("In nearby doctors")
    API_KEY = "API_KEY"
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location="+str(latitude)+"%2C"+str(longitude)+"&radius=100000&keyword=dermatologist+near+me&fields=formatted_address,name,rating,opening_hours&key="+API_KEY

    details = requests.get(url)
    details = details.json()

    names = []
    address = []
    for i in range(len(details['results'])):
        names.append(details['results'][i]['name'])
        address.append(details['results'][i]['vicinity'])
    if(len(names)>5):
        names = names[0:5]
        address = address[0:5]
    print(names)
    print(address)
    return render_template("nearby.html",names=names,address=address,length=len(names))
    

print('routes defined!')

def answer_question(question, answer_text, model, tokenizer):
    input_ids = tokenizer.encode(question, answer_text)
    print('Query has {:,} tokens.\n'.format(len(input_ids)))
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)

    start_scores, end_scores = model(tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids = tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
    #del model

    answer_start = argmax(start_scores)
    answer_end = argmax(end_scores)

    # del start_scores
    # del end_scores

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # del input_ids

    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):
        
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')
    return answer

if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True, use_reloader=False)
