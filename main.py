import ast
import keras
from keras.models import Sequential
import pickle
import tensorflow as tf
import warnings
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
    models['nn']: Sequential = keras.models.load_model('./models/sns_multinomial_ff.h5')

    # Loading Logistic Regression
    models['lr']: LogisticRegression = pickle.load(open("./models/sns_multinomial_lr.pkl", "rb"))

    # Loading BERT
    bm = __create_bert_model()
    bm.load_weights('./models/sns_bert_2.weights.h5')
    models['bert']: TFBertModel = bm


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


def main():
    load_models()
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

        input_text = input("Enter your input text: ")
        if input_text == QUIT_MESSAGE:
            exit()
        prediction_percentage = round(predict(requested_model, input_text) * 100, 2)
        print(f'Predicted likelihood for {requested_model}: {prediction_percentage}%')


if __name__ == '__main__':
    main()
