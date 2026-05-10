import gradio as gr
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved objects for the best performing model (Tuned Logistic Regression)
model_to_use = joblib.load(open("tuned_lr_model.pkl", "rb"))
vectorizer_to_use = joblib.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder_to_use = joblib.load(open("label_encoder.pkl", "rb"))

# Stopwords + Lemmatizer (simple)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def clean_text_for_prediction(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(words)

# Prediction function
def predict_with_best_model(text):
    cleaned = clean_text_for_prediction(text)
    vector = vectorizer_to_use.transform([cleaned])
    pred = model_to_use.predict(vector)
    label = label_encoder_to_use.inverse_transform(pred)
    return label[0]

# Gradio UI
app_tuned_lr = gr.Interface(
    fn=predict_with_best_model,
    inputs=gr.Textbox(lines=5, placeholder="Enter your text here..."),
    outputs="text",
    title="MBTI Personality Predictor (Tuned Logistic Regression)",
    description="Enter a text and model will predict personality type (INFJ, ENTP, etc.) using the best tuned Logistic Regression model."
)

app_tuned_lr.launch()
