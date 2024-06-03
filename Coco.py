import os
import zipfile
import requests
from pycocotools.coco import COCO
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Embedding, LSTM, Input
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from datetime import datetime
from tensorflow.keras.layers import Add,Concatenate

# Ensure TensorFlow uses the GPU if available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set the directory where you want to save the dataset files
dataDir = os.path.join(os.getcwd(), 'data')
dataType = 'train2017'
annFile = os.path.join(dataDir, 'annotations', f'captions_{dataType}.json')

# Create directories if they don't exist
os.makedirs(os.path.join(dataDir, 'annotations'), exist_ok=True)
os.makedirs(os.path.join(dataDir, 'images', dataType), exist_ok=True)

# Download COCO annotations if not already downloaded
annotations_zip_path = os.path.join(dataDir, 'annotations_trainval2017.zip')
if not os.path.exists(annFile):
    if not os.path.exists(annotations_zip_path):
        annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        print(f'Downloading {annotations_url} ...')
        response = requests.get(annotations_url)
        with open(annotations_zip_path, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded {annotations_zip_path}')

    # Extract annotations
    with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataDir)

# Ensure the annotations file exists after extraction
if not os.path.exists(annFile):
    raise FileNotFoundError(f"{annFile} not found after extraction.")

# Load annotations
coco = COCO(annFile)

# Download a subset of images and their captions if not already downloaded
ids = list(coco.anns.keys())
images = []
captions = []
for i in range(10000):
    ann_id = ids[i]
    caption = coco.anns[ann_id]['caption']
    img_id = coco.anns[ann_id]['image_id']
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info['coco_url']

    # Save the image locally
    img_path = os.path.join(dataDir, 'images', dataType, img_info['file_name'])
    if not os.path.exists(img_path):
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).resize((224, 224))
        img.save(img_path)
    img = Image.open(img_path).resize((224, 224))
    img = np.array(img)
    if len(img.shape) == 3:  # Ensure image has 3 channels
        images.append(img)
        captions.append(caption)
images = np.array(images)

# Preprocess captions
tokenizer = Tokenizer(num_words=5000, oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ')
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(captions)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Define the parameters for the synthetic data generator
batch_size = 32
height = 224
width = 224
channels = 3
seq_length = max_length  # Using max_length from the real dataset
epochs = 3


# Define the data generator function
def data_generator():
    while True:
        # Generate random image data
        images = np.random.random((batch_size, height, width, channels)).astype(np.float32)
        # Generate random input ids and labels
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_length), dtype=np.int32)
        labels = np.random.randint(0, vocab_size, (batch_size, seq_length), dtype=np.int32)

        # Yield the data in the expected format
        yield (images, input_ids), labels


# Create the TensorFlow dataset from the generator
output_signature = (
    (
        tf.TensorSpec(shape=(batch_size, height, width, channels), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, seq_length), dtype=tf.int32)
    ),
    tf.TensorSpec(shape=(batch_size, seq_length), dtype=tf.int32)
)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=output_signature
)


# Define a simple model
import os
import zipfile
import requests
from pycocotools.coco import COCO
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Embedding, LSTM, Input
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from datetime import datetime

# Ensure TensorFlow uses the GPU if available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set the directory where you want to save the dataset files
dataDir = os.path.join(os.getcwd(), 'data')
dataType = 'train2017'
annFile = os.path.join(dataDir, 'annotations', f'captions_{dataType}.json')

# Create directories if they don't exist
os.makedirs(os.path.join(dataDir, 'annotations'), exist_ok=True)
os.makedirs(os.path.join(dataDir, 'images', dataType), exist_ok=True)

# Download COCO annotations if not already downloaded
annotations_zip_path = os.path.join(dataDir, 'annotations_trainval2017.zip')
if not os.path.exists(annFile):
    if not os.path.exists(annotations_zip_path):
        annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        print(f'Downloading {annotations_url} ...')
        response = requests.get(annotations_url)
        with open(annotations_zip_path, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded {annotations_zip_path}')

    # Extract annotations
    with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataDir)

# Ensure the annotations file exists after extraction
if not os.path.exists(annFile):
    raise FileNotFoundError(f"{annFile} not found after extraction.")

# Load annotations
coco = COCO(annFile)

# Download a subset of images and their captions if not already downloaded
ids = list(coco.anns.keys())
images = []
captions = []
for i in range(10000):
    ann_id = ids[i]
    caption = coco.anns[ann_id]['caption']
    img_id = coco.anns[ann_id]['image_id']
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info['coco_url']

    # Save the image locally
    img_path = os.path.join(dataDir, 'images', dataType, img_info['file_name'])
    if not os.path.exists(img_path):
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).resize((224, 224))
        img.save(img_path)
    img = Image.open(img_path).resize((224, 224))
    img = np.array(img)
    if len(img.shape) == 3:  # Ensure image has 3 channels
        images.append(img)
        captions.append(caption)
images = np.array(images)

# Preprocess captions
tokenizer = Tokenizer(num_words=5000, oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ')
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(captions)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Define the parameters
batch_size = 32
height = 224
width = 224
channels = 3
seq_length = max_length  # Using max_length from the real dataset
embedding_dim = 256
units = 512
epochs = 3


# Define the data generator function
def data_generator():
    while True:
        # Generate random image data
        images = np.random.random((batch_size, height, width, channels)).astype(np.float32)
        # Generate random input ids and labels
        input_ids = np.random.randint(0, vocab_size, (batch_size, seq_length), dtype=np.int32)
        labels = np.random.randint(0, vocab_size, (batch_size, seq_length), dtype=np.int32)

        # Yield the data in the expected format
        yield (images, input_ids), labels


# Create the TensorFlow dataset from the generator
output_signature = (
    (
        tf.TensorSpec(shape=(batch_size, height, width, channels), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, seq_length), dtype=tf.int32)
    ),
    tf.TensorSpec(shape=(batch_size, seq_length), dtype=tf.int32)
)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=output_signature
)


# Define the encoder-decoder model

class SimpleModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        super(SimpleModel, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.embedding_dim = embedding_dim

        # Encoder (VGG16)
        self.encoder_cnn = tf.keras.applications.VGG16(include_top=False, input_shape=(height, width, channels))
        self.flatten = tf.keras.layers.Flatten()
        self.encoder_dense = tf.keras.layers.Dense(units, activation='relu')

        # Decoder
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm1 = LSTM(units, return_sequences=True, return_state=True)
        self.lstm2 = LSTM(units, return_sequences=True, return_state=True)
        self.dense_output = Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        images, input_ids = inputs

        # Encoder
        x = self.encoder_cnn(images)
        x = self.flatten(x)
        x = self.encoder_dense(x)
        x = tf.expand_dims(x, 1)
        x = tf.tile(x, [1, tf.shape(input_ids)[1], 1])  # Tile to match the sequence length

        # Decoder
        embeddings = self.embedding(input_ids)
        x = Concatenate(axis=-1)([x, embeddings])  # Concatenate to ensure compatible shapes
        x, _, _ = self.lstm1(x)
        x, _, _ = self.lstm2(x)
        output = self.dense_output(x)

        return output


# Instantiate and compile the model
model = SimpleModel(vocab_size, embedding_dim, units, batch_size)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=epochs, steps_per_epoch=10)

# Generate Captions using the real dataset (images, padded_sequences) and synthetic model
# Create directory for generated captions and images
gen_cap_dir = os.path.join(os.getcwd(), 'gen_cap')
os.makedirs(gen_cap_dir, exist_ok=True)


# Generate Captions using the real dataset (images, padded_sequences) and synthetic model
def generate_caption(image, image_index):
    # Preprocess the image
    image = np.expand_dims(image, axis=0).astype(np.float32)

    # Extract image features using the encoder part of the model
    image_features = model.encoder_cnn.predict(image)
    image_features = model.flatten(image_features)
    image_features = model.encoder_dense(image_features)
    image_features = tf.expand_dims(image_features, 1)  # Add sequence dimension

    # Initialize the caption with the start token
    caption = ['<start>']

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

        # Predict the next word
        predictions = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(predictions[0, i])

        word = tokenizer.index_word.get(yhat, '<unk>')  # Use get to avoid KeyError
        if word == '<end>':
            break
        caption.append(word)

    caption_str = ' '.join(caption)

    # Save the image and generated caption
    plt.imshow(image[0])
    plt.axis('off')
    plt.title(caption_str)
    plt.savefig(os.path.join(gen_cap_dir, f'image_{image_index}.png'))
    plt.close()

    with open(os.path.join(gen_cap_dir, f'image_{image_index}.txt'), 'w') as f:
        f.write(caption_str)

    return caption_str


for i in range(5):
    img = images[i]
    print("Generated Caption:", generate_caption(img, i))
    plt.imshow(img)
    plt.show()

# Evaluate the Model (Optional)
references = [caption.split() for caption in captions[:100]]
generated_captions = [generate_caption(img, i).split() for i, img in enumerate(images[:100])]
bleu_scores = [sentence_bleu([ref], gen) for ref, gen in zip(references, generated_captions)]
average_bleu = np.mean(bleu_scores)
print("Average BLEU Score:", average_bleu)