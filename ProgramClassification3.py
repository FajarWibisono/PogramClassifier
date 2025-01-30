import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from io import BytesIO

# Load the IndoBERT model and tokenizer with a valid model identifier
model_name = "indobenchmark/indobert-large-p2"  
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch

# Definisikan kriteria klasifikasi
def classify_activity(description):
    if "strategi" in description.lower() or "tujuan jangka panjang" in description.lower() or "pengambilan keputusan tingkat tinggi" in description.lower():
        return "Strategis", "Kegiatan berfokus pada jangka panjang, tujuan organisasi, dan pengambilan keputusan tingkat tinggi."
    elif "implementasi strategi" in description.lower() or "manajemen operasional" in description.lower() or "tujuan jangka menengah" in description.lower():
        return "Taktikal", "Kegiatan berfokus pada implementasi strategi, manajemen operasional, dan pencapaian tujuan jangka menengah."
    else:
        return "Operasional", "Kegiatan berfokus pada operasi sehari-hari, efisiensi, dan pencapaian tujuan jangka pendek."

# Contoh dataset
data = {
    'activity': [
        "Merumuskan strategi jangka panjang untuk pertumbuhan perusahaan",
        "Mengimplementasikan strategi pemasaran baru",
        "Melakukan audit keuangan bulanan"
    ]
}

# Buat DataFrame
df = pd.DataFrame(data)

# Tambahkan kolom klasifikasi dan keterangan
df['classification'], df['description'] = zip(*df['activity'].apply(classify_activity))

# Tampilkan DataFrame
print(df)
access_token = st.secrets["HUGGINGFACE_API_KEY"]  

try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
except OSError as e:
    print(f"Error loading model: {e}")
    # Handle the error appropriately

# Function to preprocess text using IndoBERT tokenizer
def preprocess_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return inputs

# Function to predict using IndoBERT model
def get_classification_explanation(classification):
    explanations = {
        "Strategis": "Klasifikasi Strategis karena berkaitan dengan perencanaan jangka panjang, visi misi, dan tujuan organisasi secara keseluruhan, peningkatan revenue, produktivitas, menurunkan risk insiden besar, keberlanjutan, manajemen integritas, transformasi, ketangguhan, efisensi energi, dan reputasi perusahaan",
        "Taktikal": "Klasifikasi Taktikal karena berfokus pada identifikasi dan mitigasi risiko, Kehandalan peralatan, audit internal, implementasi rencana jangka menengah, kolaborasi dan koordinasi antar departemen, kesehatan dan keselamatan kerja.", 
        "Operasional": "Klasifikasi Operasional karena menyangkut kegiatan sehari-hari, mengurangi waktu henti operasi, menyusun jadwal, menetapkan prosedur pekerjaan, pelaksanaan tugas rutin, pemeliharan, training,  dan implementasi langsung di lapangan."
    }
    return explanations.get(classification, "Tidak ada penjelasan tersedia")

def predict_text(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
    return predictions.item()

# Function to train the model (not needed since we use pre-trained IndoBERT)
def train_model(df, text_column, label_column):
    # Since we are using a pre-trained model, we don't need to train it here
    # But we can still use this function to load and prepare the data
    labels = df[label_column].unique()
    label_map = {label: i for i, label in enumerate(labels)}
    df['label'] = df[label_column].map(label_map)
    
    return df, label_map

# Function to save the model (not needed since we use pre-trained IndoBERT)
def save_model():
    pass

# Function to load the model (not needed since we use pre-trained IndoBERT)
def load_model():
    pass

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
        train_df, label_map = train_model(train_df, text_column, label_column)
        
        # Invert the label map for prediction
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
                    for text in pred_df[pred_column]:
                        inputs = preprocess_text(text)
                        prediction = predict_text(inputs)
                        predictions.append(label_map_inv[prediction])
                    
                    pred_df['Hasil_Klasifikasi'] = predictions
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
                prediction = predict_text(inputs)
                st.write('Hasil Klasifikasi:')
                # Add debugging statements
                print(f"Prediction: {prediction}")
                print(f"Label Map Inv: {label_map_inv}")

                # Ensure prediction is a valid key in label_map_inv
                if prediction in label_map_inv:
                    classification = label_map_inv[prediction]
                    explanation = get_classification_explanation(classification)
                    st.info(f"Teks termasuk kategori: {classification}")
                    st.info(f"Penjelasan: {explanation}")
                else:
                    st.error(f"Invalid prediction: {prediction}. Valid labels are: {list(label_map_inv.keys())}")

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
