import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
MODEL_PATH = "D:/Medical Experts/skin_disease_model_mobilenet.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Disease classes and expert knowledge
classes = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion']

knowledge_base = {
    'Actinic keratosis': "Rough, scaly patches on the skin, often due to prolonged sun exposure. It is considered precancerous and can develop into squamous cell carcinoma if untreated. Protect the skin from sun exposure using sunscreen and wear protective clothing. Consult a dermatologist for potential treatments like cryotherapy or topical medications.",
    'Atopic Dermatitis': "A chronic condition causing itchy, inflamed skin, often triggered by allergens, stress, or irritants. Management includes moisturizing frequently, using anti-inflammatory creams, and avoiding known triggers. Severe cases may require prescription treatments like corticosteroids or biologics. Regular dermatological follow-ups are recommended.",
    'Benign keratosis': "Non-cancerous skin growths that may appear scaly, rough, or warty. These are typically harmless and can be left untreated unless they cause discomfort. Treatment options include cryotherapy or laser removal for cosmetic concerns. Regular monitoring is advised to ensure no malignant changes.",
    'Dermatofibroma': "Small, firm, raised nodules that often occur on the legs or arms. These are benign and usually painless, but they may itch or become tender. No treatment is typically needed unless symptomatic. Surgical removal is an option for persistent cases.",
    'Melanocytic nevus': "Commonly referred to as moles, these are generally harmless but should be monitored for changes in size, shape, or color. Use the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolving) to check for potential malignancy. Regular skin checks with a dermatologist are recommended.",
    'Melanoma': "A dangerous form of skin cancer that can spread rapidly. Early detection is crucial and involves identifying new or changing moles using the ABCDE rule. Seek immediate consultation with a dermatologist or oncologist for evaluation and treatment options such as surgical excision, immunotherapy, or targeted therapy.",
    'Squamous cell carcinoma': "A type of skin cancer that often presents as a red, scaly patch, open sore, or wart-like growth. It can be aggressive if untreated. Early medical intervention, including biopsy, surgical removal, or radiation therapy, is critical. Protect the skin from further sun damage.",
    'Tinea Ringworm Candidiasis': "A fungal infection characterized by ring-shaped, scaly rashes with clear centers. Common in warm, moist environments. Treatment includes topical antifungal creams, oral antifungal medications for severe cases, and maintaining hygiene to prevent recurrence.",
    'Vascular lesion': "Abnormalities in blood vessels that can appear as red, blue, or purple marks on the skin. Types include hemangiomas, spider veins, or port-wine stains. Treatment options range from observation to laser therapy or surgical interventions, depending on the type and severity. Consultation with a specialist is advised."
}

def preprocess_image(image):
    """
    Preprocess the uploaded image for model prediction.
    """
    image = image.resize((240, 240))
    image = np.asarray(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)

# Streamlit Web Interface
st.title("Skin Disease Classification and Recommendations")
st.write("Upload an image to classify the skin condition and get recommendations.")

# File uploader
uploaded_file = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = classes[predicted_class]

    # Display prediction
    st.write(f"### Predicted Class: {predicted_label}")

    # Display expert knowledge
    st.write(f"### Recommendations / Symptoms:")
    st.write(knowledge_base[predicted_label])

st.write("\n---")
st.write("Disclaimer: This tool is for informational purposes only and not a substitute for professional medical advice.")
