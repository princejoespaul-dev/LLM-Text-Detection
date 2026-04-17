import tensorflow as tf
import pickle

class PredictionPipeline:
    def __init__(self):
        self.model = tf.keras.models.load_model("assests/text_classification_model.h5")
        self.vectorizer = pickle.load(open("assests/tfidf_tokenizer.pkl", "rb"))

    def predict(self, text):
        transformed = self.vectorizer.transform([text]).toarray()
        prediction = self.model.predict(transformed)

        return "AI Generated" if prediction[0][0] > 0.5 else "Human Written"

         