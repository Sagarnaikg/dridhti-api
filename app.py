from flask import Flask, request
import cv2
import numpy as np
from keras.layers import Dense,  LSTM, TimeDistributed, Embedding, Activation, RepeatVector, Concatenate
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.resnet import ResNet50
from tqdm import tqdm

# logic part

vocab = np.load('vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v: k for k, v in vocab.items()}


embedding_size = 128
vocab_size = len(vocab)
max_len = 40


image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))


language_model = Sequential()


language_model.add(Embedding(input_dim=vocab_size,
                   output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs=out)

model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights.h5')
incept_model = ResNet50(include_top=True)
# take the 2nd last layer
last = incept_model.layers[-2].output
cnn_model = Model(inputs=incept_model.input, outputs=last)
###

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'


@app.route("/")
def base():
    return {"message": "api is live"}


@ app.route('/upload', methods=['POST', 'GET'])
def upload_file():

    if 'image' not in request.files:
        return {'message': 'No file part in the request'}

    global model, vocab, inv_vocab

    img = request.files['image']
    img.save('static/image.jpg')

    image = cv2.imread('static/image.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image = np.reshape(image, (1, 224, 224, 3))

    incept = cnn_model.predict(image).reshape(1, 2048)
    text_in = ['startofseq']

    final = ''

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences(
            [encoded], maxlen=max_len, padding='post', truncating='post').reshape(1, max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)

    print(final)
    return {"result": final}
