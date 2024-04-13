import ast
import keras
from keras.models import Sequential
import pickle
import tensorflow as tf
import warnings
import keras_tuner as kt
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import TFBertModel, BertTokenizer, logging

warnings.filterwarnings('ignore')
logging.set_verbosity_error()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=False, return_dict=False)
bert_model.trainable = False

QUIT_MESSAGE = '!quit'
models = {}

vocabulary = ast.literal_eval(open("./datasets/processed/vocabulary.txt", "r").read())
v = CountVectorizer(input='content', stop_words='english', binary=False, vocabulary=vocabulary, tokenizer=None,
                    preprocessor=None)


def load_models():
    # Loading Feedforward Neural Network
    models['nn']: Sequential = keras.models.load_model('./models/sns_tuned_nn.h5')

    # Loading Logistic Regression
    models['lr']: LogisticRegression = pickle.load(open("./models/sns_best_lr.pkl", "rb"))

    # Loading BERT
    with open('./models/sns_tuned_bert_hp_config.json', 'r') as f:
        hp_config = json.loads(f.read())
        f.close()
    hp = kt.HyperParameters().from_config(hp_config)
    tuned_bert = __create_tuned_bert_model(hp)
    tuned_bert.load_weights('./models/sns_tuned_bert.weights.h5')
    models['bert']: TFBertModel = tuned_bert

    # bm = __create_bert_model()
    # bm.load_weights('./models/sns_bert_2.weights.h5')
    # models['bert']: TFBertModel = bm


def predict(model: str, text: str) -> float:
    """
    Predicts inflammatory and/or offensiveness of textual content (supplied string)
    Args:
        model:
        text:

    Returns:

    """
    # Vectorize the input text
    X_input = v.transform([text])
    if model == 'nn':
        # Get the prediction from the trained model
        prediction = models[model].predict(X_input.toarray(), verbose=0)

        # The output is a probability value between 0 and 1
        inflammatory_level = prediction[0][0]

        return inflammatory_level
    if model == 'lr':
        return float(models[model].predict_proba(X_input)[:, 1])
    if model == 'bert':
        input_encoded = tokenizer.batch_encode_plus(
            [text], max_length=128, padding='max_length', truncation=True, return_tensors='tf')
        input_batch = [input_encoded['input_ids'], input_encoded['attention_mask'], input_encoded['token_type_ids']]
        pred = models[model].predict(input_batch, verbose=0)
        return pred[0][0]


def __create_bert_model():
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    token_type_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='token_type_ids')

    bert_output = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    sequence_output = bert_output[0]
    # Take the [CLS] token representation
    pooled_output = sequence_output[:, 0, :]

    x = tf.keras.layers.Dropout(0.2)(pooled_output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def __create_tuned_bert_model(hp: kt.HyperParameters):
    MAX_LENGTH = 128
    input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')
    token_type_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='token_type_ids')

    bert_output = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    sequence_output = bert_output[0]
    # Take the [CLS] token representation
    pooled_output = sequence_output[:, 0, :]

    dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=64)

    x = tf.keras.layers.Dropout(dropout_rate)(pooled_output)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    learning_rate = hp.Choice('learning_rate', values=[1e-5, 1e-4, 1e-3])

    model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    load_models()
    print('\n' * 10)
    print('Type \'nn\' for feedforward neural network model.\n'
          'Type \'lr\' for logistic regression.\n'
          'Type \'bert\' for BERT model.\n\n'
          f'(Type \'{QUIT_MESSAGE}\' to quit anytime.)\n')

    while True:
        requested_model = input('Enter model name: ').strip().lower()

        if requested_model == QUIT_MESSAGE:
            exit()
        elif requested_model not in models:
            print('Invalid model input!')
            continue

        input_text = input("Enter your input text: ").strip()
        if input_text == QUIT_MESSAGE:
            exit()
        prediction_percentage = round(predict(requested_model, input_text) * 100, 2)
        print(f'Predicted likelihood for {requested_model}: {prediction_percentage}%')
        print('\n')


if __name__ == '__main__':
    main()
