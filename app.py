from word_embedding.word2vec_gensim import Word2Vec
from text_classification.short_text_classifiers import BiDirectionalLSTMClassifier
from tokenization.crf_tokenizer import CrfTokenizer
import flask
import pandas as pd

app = flask.Flask(__name__)
word2vec_model = Word2Vec.load('models/pretrained_word2vec.bin')

tokenizer = CrfTokenizer(config_root_path='tokenization/',
                         model_path='models/pretrained_tokenizer.crfsuite')
model = BiDirectionalLSTMClassifier(tokenizer=tokenizer, word2vec=word2vec_model.wv,
                                    model_path='models/app.h5',
                                    n_class=3)
label_dict = {0: 'mo_vnexpress', 1: 'mo_dantri', 2: 'mo_truyenfull'}


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if params is None:
        params = flask.request.args

    # if parameters are found, return a prediction
    if params is not None:
        x = pd.DataFrame.from_dict(params, orient='index').to_numpy(dtype=str).tolist()
        data["prediction"] = model.classify(sentences=x[0], label_dict=label_dict)[0]

    # return a response in json format
    return flask.jsonify(data)


# start the flask app, allow remote connections
app.run(host='0.0.0.0')
