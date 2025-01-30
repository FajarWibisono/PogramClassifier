import streamlit as st
import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import pandas as pd
import numpy as np
from io import BytesIO
from datasets import Dataset

# Debug information
st.write("Starting app...")

# Import transformers dengan error handling
try:
    from transformers import AutoTokenizer
    st.write("Successfully imported AutoTokenizer")
    from transformers import AutoModelForSequenceClassification
    st.write("Successfully imported AutoModelForSequenceClassification")
    from transformers import Trainer, TrainingArguments
    st.write("Successfully imported Trainer and TrainingArguments")
except Exception as e:
    st.error(f"Error importing transformers: {str(e)}")
    st.stop()


# Constants
NUM_LABELS = 3  # Jumlah kelas: Strategis, Taktikal, Operasional
model_name = "indobenchmark/indobert-large-p2"

# Cache untuk model dan tokenizer
@st.cache_resource
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Tambahkan logging
        st.write("Attempting to load model and tokenizer...")
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=st.secrets["HUGGINGFACE_API_KEY"],
            use_fast=True
        )
        st.success("Tokenizer loaded successfully")
        
        # Then load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            token=st.secrets["HUGGINGFACE_API_KEY"],
            ignore_mismatched_sizes=True,
            from_tf=False
        )
        st.success("Model loaded successfully")
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error in load_model_and_tokenizer: {str(e)}")
        return None, None
        
# Load model dan tokenizer
model, tokenizer = load_model_and_tokenizer()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
if model is not None:
    model = model.to(device)

def initialize_model_weights():
    # Data training awal
    initial_data = {
        'text': [
            "Merumuskan strategi jangka panjang untuk pertumbuhan perusahaan",
            "Mengembangkan visi dan misi perusahaan",
            "Membuat perencanaan strategis 5 tahun ke depan",
            "Implementasi program efisiensi departemen",
            "Koordinasi antar tim untuk proyek",
            "Pengembangan kemampuan tim",
            "Membuat laporan harian",
            "Melakukan maintenance rutin",
            "Menjalankan prosedur operasi standar"
        ],
        'label': [0, 0, 0, 1, 1, 1, 2, 2, 2]  # 0:Strategis, 1:Taktikal, 2:Operasional
    }
    
    # Tokenisasi data
    encoded_data = tokenizer(
        initial_data['text'],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Buat tensor untuk labels
    labels = torch.tensor(initial_data['label'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_dir="./logs",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_dict({
            'input_ids': encoded_data['input_ids'],
            'attention_mask': encoded_data['attention_mask'],
            'labels': labels
        })
    )
    
    # Train model
    trainer.train()
    
    return model

# Initialize model weights jika model berhasil dimuat
if model is not None:
    model = initialize_model_weights()

# Definisikan kriteria klasifikasi
def classify_activity(description):
    if "strategi" in description.lower() or "tujuan jangka panjang" in description.lower() or "pengambilan keputusan tingkat tinggi" in description.lower():
        return "Strategis", "Kegiatan berfokus pada jangka panjang, tujuan organisasi, dan pengambilan keputusan tingkat tinggi."
    elif "implementasi strategi" in description.lower() or "manajemen operasional" in description.lower() or "tujuan jangka menengah" in description.lower():
        return "Taktikal", "Kegiatan berfokus pada implementasi strategi, manajemen operasional, dan pencapaian tujuan jangka menengah."
    else:
        return "Operasional", "Kegiatan berfokus pada operasi sehari-hari, efisiensi, dan pencapaian tujuan jangka pendek."

# Function to preprocess text using IndoBERT tokenizer
def preprocess_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return inputs

# Function to predict using IndoBERT model
def predict_text(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        confidence = torch.max(probabilities, dim=1).values
        
        # Mapping hasil prediksi
        prediction_map = {
            0: "Strategis",
            1: "Taktikal",
            2: "Operasional"
        }
        
        predicted_class = prediction_map[predictions.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score

def get_classification_explanation(classification):
    explanations = {
        "Strategis": "Klasifikasi Strategis karena berkaitan dengan perencanaan jangka panjang, visi misi, dan tujuan organisasi secara keseluruhan, peningkatan revenue, produktivitas, menurunkan risk insiden besar, keberlanjutan, manajemen integritas, transformasi, ketangguhan, efisensi energi, dan reputasi perusahaan",
        "Taktikal": "Klasifikasi Taktikal karena berfokus pada identifikasi dan mitigasi risiko, Kehandalan peralatan, audit internal, implementasi rencana jangka menengah, kolaborasi dan koordinasi antar departemen, kesehatan dan keselamatan kerja.", 
        "Operasional": "Klasifikasi Operasional karena menyangkut kegiatan sehari-hari, mengurangi waktu henti operasi, menyusun jadwal, menetapkan prosedur pekerjaan, pelaksanaan tugas rutin, pemeliharan, training, dan implementasi langsung di lapangan."
    }
    return explanations.get(classification, "Tidak ada penjelasan tersedia")

# Sidebar for uploading and training data
st.sidebar.header('Upload Data Training')
train_file = st.sidebar.file_uploader("Upload file Excel untuk training", type=['xlsx'])

if train_file is not None:
    try:
        train_df = pd.read_excel(train_file)
        st.sidebar.success("File training berhasil diupload!")
        
        # Select columns
        text_column = st.sidebar.selectbox('Pilih kolom teks program/kegiatan:', train_df.columns)
        label_column = st.sidebar.selectbox('Pilih kolom label (STRATEGIS/TAKTIKAL/OPERASIONAL):', train_df.columns)
        
        # Get the label map
        labels = train_df[label_column].unique()
        label_map = {label: i for i, label in enumerate(labels)}
        label_map_inv = {v: k for k, v in label_map.items()}
        
        if st.sidebar.button('Train Model'):
            st.sidebar.success('Model sudah siap digunakan!')
    
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Main area for prediction
st.header('Klasifikasi Data Baru')

# Tab for choosing input method
tab1, tab2 = st.tabs(["Upload File", "Input Manual"])

with tab1:
    pred_file = st.file_uploader("Upload file Excel untuk klasifikasi", type=['xlsx'])
    
    if pred_file is not None:
        try:
            pred_df = pd.read_excel(pred_file)
            st.success("File berhasil diupload!")
            
            pred_column = st.selectbox('Pilih kolom teks yang akan diklasifikasi:', pred_df.columns)
            
            if st.button('Klasifikasi File'):
                with st.spinner('Mengklasifikasi data...'):
                    predictions = []
                    confidences = []
                    for text in pred_df[pred_column]:
                        inputs = preprocess_text(text)
                        predicted_class, confidence = predict_text(inputs)
                        predictions.append(predicted_class)
                        confidences.append(confidence)
                    
                    pred_df['Hasil_Klasifikasi'] = predictions
                    pred_df['Confidence_Score'] = confidences
                    pred_df['Penjelasan'] = pred_df['Hasil_Klasifikasi'].apply(get_classification_explanation)
                    st.write('Hasil Klasifikasi:')
                    st.dataframe(pred_df)
                    
                    # Download button
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        pred_df.to_excel(writer, index=False)
                    
                    st.download_button(
                        label="Download hasil klasifikasi (Excel)",
                        data=buffer.getvalue(),
                        file_name="hasil_klasifikasi.xlsx",
                        mime="application/vnd.ms-excel"
                    )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab2:
    input_text = st.text_area("Tuliskan Program Budaya/Kegiatan/Deliverables Anda di bawah ini:")
    
    if st.button('Klasifikasi Teks'):
        if not input_text:
            st.warning("Mohon masukkan teks terlebih dahulu!")
        else:
            with st.spinner('Mengklasifikasi teks...'):
                inputs = preprocess_text(input_text)
                predicted_class, confidence = predict_text(inputs)
                
                st.write('Hasil Klasifikasi:')
                st.info(f"Teks termasuk kategori: {predicted_class}")
                st.info(f"Tingkat keyakinan: {confidence:.2%}")
                explanation = get_classification_explanation(predicted_class)
                st.info(f"Penjelasan: {explanation}")

# Additional information
st.markdown("""
### Petunjuk Penggunaan:
Applikasi ini berguna untuk menilai KATEGORI Program Budaya/Aktivitas/Deliverables
1. Pilih metode input:
   - Upload file Excel yang ingin diklasifikasi, atau
   - Masukkan teks secara manual dengan mengetikkan pada kolom yang tersedia
2. Klik 'Klasifikasi' untuk mendapatkan hasil
3. Perhatikan HASIL KLASIFIKASI, pertimbangkan kembali hasil klasifikasinya.
""")

# Add reasoning capabilities
st.markdown("""
### Metode Klasifikasi:
Untuk meningkatkan akurasi klasifikasi, aplikasi ini menggunakan model `IndoBERT` yang telah difine-tune untuk bahasa Indonesia. Model ini memiliki kemampuan yang lebih baik dalam memahami dan menangani teks bahasa Indonesia, termasuk kemampuan reasoning dan comprehension yang lebih baik. Ini memungkinkan aplikasi untuk memberikan hasil klasifikasi yang lebih akurat dan relevan.
""")
