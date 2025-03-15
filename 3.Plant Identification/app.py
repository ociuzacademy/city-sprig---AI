import json
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore


from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

IMAGE_SIZE = [224,224]
vgg = VGG16(input_shape=IMAGE_SIZE +[3] , weights = 'imagenet' , include_top = False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
class_names = ['Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avacado', 'Bamboo', 'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon', 'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Pappaya', 'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel']
prediction = Dense(len(class_names) , activation = "softmax")(x)
model = Model(inputs = vgg.input,outputs = prediction)






# Load the model
#model = load_model('my_model.h5', compile=False)
model.load_weights('my_model.h5')


# Define the plant names
names = [
    'Aloe Vera', 'Amla', 'Amruta Balli', 'Arali', 'Ashoka', 'Ashwagandha',
    'Avocado', 'Bamboo', 'Basale', 'Betel', 'Betel Nut', 'Brahmi', 'Castor',
    'Curry Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Guava', 'Geranium',
    'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon',
    'Lemon Grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa',
    'Nooni', 'Papaya', 'Pepper', 'Pomegranate', 'Raktachandini',
    'Rose', 'Sapota', 'Tulsi', 'Wood Sorel'
]

# Preprocess image function using OpenCV
def preprocess_image(image):
    # Decode the image
    #image_array = np.frombuffer(image, np.uint8)
    #img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Resize to 224x224
    img_resized = cv2.resize(image, (224, 224))
    
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Expand dimensions to match model input shape
    return np.expand_dims(img_rgb, axis=0)


def get_plant_info(plant_name):
    # Load the JSON data from the file
    with open('plants.json', 'r') as file:
        data = json.load(file)

    # Search for the plant in the data
    for plant in data['plants']:
        if plant['name'].lower() == plant_name.lower():
            return plant['description'], plant['uses']

    return None, None




################################################## Read test image, edit the image name to test more images in the folder ###################################################################
image_path = 'test_images/Aloevera/4291.jpg'




image = cv2.imread(image_path)
print(image.shape)
processed_image = preprocess_image(image)
        

# Make prediction
prediction = model.predict(processed_image)
predicted_class = np.argmax(prediction, axis=1)
        
# Convert to one-hot encoded format
one_hot_encoded = np.zeros(len(names))
one_hot_encoded[predicted_class] = 1


description, uses = get_plant_info(names[predicted_class[0]])
print('Identified Plant: ', names[predicted_class[0]])
print('Description: ', description)
print('Uses: ', uses)