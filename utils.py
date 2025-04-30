import json
import glob , os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

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
    
class MModel:
    def __init__(self, name):
        self.name = name
        self.num_params = None
        self.model = None
        self.history = None
        
    def set_model(self, model):
        self.model = model
        self.num_params = self.model.count_params()
        
    def summary(self):
        self.model.summary()
        
    def compile(self, **kwargs):
        self.model.compile(**kwargs)
    
    def model_path(self):
        return f"models/{self.name}_{self.num_params}/"
    
    def _fit(self, **kwargs):
        return self.model.fit(self.gen_train,validation_data=self.gen_test, **kwargs)
        
    def fit(self, cache=True, continue_training=False, **kwargs):
        input_size = self.model.input_shape[1:3]
        print(input_size)
        self.gen_train = generator_transformations("train", input_size)
        self.gen_test = generator("test", input_size)
        model_path = self.model_path()
        model_file = f"{model_path}{self.name}.h5"
        model_history_file = f"{model_path}{self.name}_history.json"
        if cache and os.path.exists(model_file) and os.path.exists(model_history_file):
            self.model = tf.keras.models.load_model(model_file)
            with open(model_history_file, "r") as f:
                self.history = json.load(f)
            if continue_training:
                self.history = self._fit(**kwargs).history
                with open(model_history_file, "w") as f:
                    json.dump(self.history, f)
                self.model.save(model_file)
            return self.history
        
        os.makedirs(model_path, exist_ok=True)
        self.history = self._fit(**kwargs).history
        self.model.save(model_file)
        with open(model_history_file, "w") as f:
            json.dump(self.history, f)
        return self.history
        
    def evaluate(self):        
        # accuracy and loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='accuracy')
        plt.plot(self.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='loss')
        plt.plot(self.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        
        plt.savefig(f"models/{self.name}_{self.num_params}/accuracy_loss.png")
        
        # confusion matrix
        self.gen_test.reset()
        self.gen_test.samples = self.gen_test.n
        self.gen_test.batch_size = self.gen_test.n
        X_test, y_test = next(self.gen_test)
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f"models/{self.name}_{self.num_params}/confusion_matrix.png")
          
        # classification report save to file the output
        with open(f"models/{self.name}_{self.num_params}/classification_report.txt", "w") as f:
            f.write(classification_report(y_true, y_pred))
            f.close()
        
# test_generator.reset()
# # use all data in the test set
# test_generator.samples = test_generator.n
# test_generator.batch_size = test_generator.n
# X_test , y_test = next(test_generator)
# print(X_test.shape)
# y_pred = model1.predict(X_test)
# y_pred = np.argmax(y_pred, axis=1)
# y_test = np.argmax(y_test, axis=1)