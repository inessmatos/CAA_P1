import glob , os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images(base_path:str):
    path = os.path.join(base_path, "*/*.jpg")
    return glob.glob(path)

def one_hot_encode(y):
    label_dict = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4 , 'trash': 5}
    y = np.array([label_dict[label] for label in y])
    return np.eye(6)[y]

def generator_transformations(path:str,size:tuple[int,int]=(200,200)):
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True, 
    )
    
    return datagen.flow_from_directory(
        directory=path,
        target_size=size,
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )
     
def generator(path:str,size:tuple[int,int]=(200,200)):
    datagen = ImageDataGenerator(
        rescale=1./255,
    )
    
    return datagen.flow_from_directory(
        directory=path,
        target_size=size,
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )