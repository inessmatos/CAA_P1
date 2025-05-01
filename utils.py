import json
import glob , os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import ViTFeatureExtractor
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
            
def infinite_gen(seq):
    while True:
        for i in range(len(seq)):
            yield seq[i]
        seq.on_epoch_end() 
        
def wrap_with_dataset(generator_wrapper):
    output_signature = generator_wrapper.get_output_signature()
    return tf.data.Dataset.from_generator(lambda: infinite_gen(generator_wrapper), output_signature=output_signature)

class ViTDataGeneratorWrapper(tf.keras.utils.Sequence): # Or your base class

    def __init__(self, generator, feature_extractor):
        self.generator = generator
        self.feature_extractor = feature_extractor
        self.class_indices = self.generator.class_indices # Store class indices mapping

    def __getitem__(self, index):
        X, y = self.generator.__getitem__(index)
        features = self.feature_extractor(X, return_tensors="tf")
        y = tf.convert_to_tensor(y) 
        return {"pixel_values": features["pixel_values"]}, y
    
    def __len__(self):
        return len(self.generator)
    
    def reset(self):
        self.generator.reset()
        
    def on_epoch_end(self):
        self.generator.on_epoch_end()
        
    def get_output_signature(self):
        X_sample, y_sample = self.__getitem__(0)
        sig_X = {
            name: tf.TensorSpec(shape=(None, *tensor.shape[1:]), dtype=tensor.dtype, name=name)
            for name, tensor in X_sample.items()
        }
        sig_y = tf.TensorSpec(shape=(None,) + tuple(y_sample.shape)[1:], dtype=y_sample.dtype, name="labels")
        return sig_X, sig_y
          
class LModel:
    def __init__(self, name):
        self.name = name
        self.num_params = None
        self.model: tf.keras.Model = None
        self.feature_extractor: ViTFeatureExtractor = None # Added
        self.history = None
        self.gen_train_wrapper: ViTDataGeneratorWrapper = None # Added
        self.gen_test_wrapper: ViTDataGeneratorWrapper = None # Added
        self.base_gen_train = None # Keep track of underlying generator
        self.base_gen_test = None # Keep track of underlying generator
        self.class_indices = None # Store class indices mapping

    def set_model(self, model, feature_extractor):
        """Sets the model and its corresponding feature extractor."""
        self.model = model
        self.feature_extractor = feature_extractor
        self.num_params = self.model.count_params()
        print(f"Model '{self.name}' set with {self.num_params} parameters.")
        print(f"Feature extractor requires input size: {self.feature_extractor.size}")

    def summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Model not set. Call set_model() first.")

    def compile(self, **kwargs):
        """Compiles the Keras model."""
        if self.model:
            print("Compiling model...")
            self.model.compile(**kwargs)
            print("Model compiled.")
        else:
            print("Model not set. Call set_model() and compile() first.")

    def model_path(self, base_save_dir="models"):
        """Generates the path for saving model artifacts."""
        if self.num_params is None:
             # Use a default if params not counted yet, though set_model should handle this
             path_name = f"{self.name}_unknown_params"
        else:
             path_name = f"{self.name}_{self.num_params}"
        return os.path.join(base_save_dir, path_name)

    def _fit(self, train_seq, valid_seq, epochs, initial_epoch=0, steps_per_epoch=None, validation_steps=None):
        """Internal fit method using Keras Sequence objects directly."""
        print(f"Starting training via _fit using Sequence for {epochs} epochs...")
        history = self.model.fit(
            train_seq,
            validation_data=valid_seq,
            epochs=epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )
        print("Training via _fit finished.")
        return history

    def fit(self, train_path: str, test_path: str, batch_size: int, epochs: int,
            model_save_dir="models", cache=True, continue_training=False):
        """
        Sets up generators, wraps them, and trains the model—no tf.data.Dataset involved.
        """
        # 1. Ensure model & feature extractor
        if not self.model or not self.feature_extractor:
            print("Model and feature extractor must be set before fitting. Call set_model().")
            return None
        if not self.model.optimizer:
            print("Model must be compiled before fitting. Call compile().")
            return None

        # 2. Create base ImageDataGenerators
        input_size = (self.feature_extractor.size['height'], self.feature_extractor.size['width'])
        base_train = generator_transformations(train_path, input_size)
        base_test  = generator(test_path, input_size)
        self.class_indices = base_train.class_indices
        print(f"Class Indices: {self.class_indices}")

        # 3. Wrap with your ViTDataGeneratorWrapper
        train_seq = ViTDataGeneratorWrapper(base_train, self.feature_extractor)
        test_seq  = ViTDataGeneratorWrapper(base_test,  self.feature_extractor)

        # 4. Prepare caching logic (unchanged)…
        _model_path = self.model_path(model_save_dir)
        os.makedirs(_model_path, exist_ok=True)
        model_file         = os.path.join(_model_path, f"{self.name}.keras")
        history_file       = os.path.join(_model_path, f"{self.name}_history.json")
        class_indices_file = os.path.join(_model_path, f"{self.name}_class_indices.json")

        start_epoch = 0
        loaded_history = {}

        if cache and os.path.exists(model_file) and os.path.exists(history_file):
            # load cached model + history…
            # (same as before)
            if continue_training:
                # continue training from start_epoch…
                train_ds = wrap_with_dataset(train_seq)
                valid_ds = wrap_with_dataset(test_seq)
                train_steps = len(train_seq)  # Compute from the Sequence, before wrapping
                valid_steps = len(test_seq)
                hist_obj = self._fit(train_ds, valid_ds, start_epoch + epochs, initial_epoch=start_epoch,steps_per_epoch=train_steps, validation_steps=valid_steps)
                # merge & save…
                return self.history
            else:
                print("Using cached model and history without further training.")
                return loaded_history

        # 5. Train from scratch
        print("No cache found or cache=False. Training from scratch...")
        self.gen_train_wrapper = train_seq
        self.gen_test_wrapper = test_seq
        train_ds = wrap_with_dataset(train_seq)
        valid_ds = wrap_with_dataset(test_seq)
        train_steps = len(train_seq)  # Compute from the Sequence, before wrapping
        valid_steps = len(test_seq)
        hist_obj = self._fit(train_ds, valid_ds, start_epoch + epochs, initial_epoch=start_epoch,steps_per_epoch=train_steps, validation_steps=valid_steps)
        if hist_obj:
            self.history = hist_obj.history
            print("Saving new model and history...")
            self.model.save(model_file)
            with open(history_file, "w") as f: json.dump(self.history, f)
            with open(class_indices_file, "w") as f: json.dump(self.class_indices, f)
            return self.history
        else:
            print("Training failed.")
            return None

    def evaluate(self, model_save_dir="models"):
        """Evaluates the model using the test generator and saves plots/reports."""
        if not self.model or not self.history:
            print("Model not trained or history not available. Call fit() first.")
            return
        if not self.gen_test_wrapper:
             print("Test generator not available. Call fit() first.")
             return

        _model_path = self.model_path(model_save_dir)
        print(f"Evaluating model. Saving artifacts to: {_model_path}")

        # --- Plot Accuracy and Loss ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(self.history['loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend(loc='upper right')

        plt.tight_layout()
        acc_loss_plot_file = os.path.join(_model_path, "accuracy_loss.png")
        plt.savefig(acc_loss_plot_file)
        print(f"Saved accuracy/loss plot to: {acc_loss_plot_file}")
        plt.close() # Close the plot to free memory

        # --- Confusion Matrix and Classification Report ---
        print("Generating predictions for evaluation...")
        # Need to iterate through the test wrapper to get all predictions
        y_pred_logits = []
        y_true = []
        num_test_batches = len(self.gen_test_wrapper)

        for i in range(num_test_batches):
             print(f"  Predicting batch {i+1}/{num_test_batches}")
             pixel_values_batch, labels_batch = self.gen_test_wrapper[i]
             batch_preds = self.model.predict(pixel_values_batch)
             # The model output might be logits or probabilities in a dataclass structure
             if hasattr(batch_preds, 'logits'):
                 y_pred_logits.append(batch_preds.logits.numpy()) # Extract logits if necessary
             else:
                 y_pred_logits.append(batch_preds) # Assume direct output is logits/probs

             y_true.append(np.argmax(labels_batch, axis=1)) # Convert one-hot back to integer labels

        y_pred_logits = np.concatenate(y_pred_logits, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.argmax(y_pred_logits, axis=1) # Get predicted class index

        # Ensure we have class labels for the report and matrix
        if self.class_indices:
             # Sort labels by index (0, 1, 2...)
             labels = [k for k, v in sorted(self.class_indices.items(), key=lambda item: item[1])]
        else:
             # Fallback if class indices weren't loaded/saved
             num_classes_found = len(np.unique(y_true))
             labels = [f"Class_{i}" for i in range(num_classes_found)]
             print("Warning: Class indices not found, using generic labels.")


        # Confusion Matrix
        cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
        plt.figure(figsize=(10, 8)) # Adjusted size
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_plot_file = os.path.join(_model_path, "confusion_matrix.png")
        plt.savefig(cm_plot_file)
        print(f"Saved confusion matrix plot to: {cm_plot_file}")
        plt.close()

        # Classification Report
        report = classification_report(y_true, y_pred, target_names=labels)
        print("\nClassification Report:")
        print(report)
        report_file = os.path.join(_model_path, "classification_report.txt")
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Saved classification report to: {report_file}")