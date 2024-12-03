import streamlit as st
import joblib  # Untuk memuat model dan scaler

# Function to load the trained model and scaler
def load_model_and_scaler():
    model = joblib.load('mental_health_model.pkl')  # Memuat model yang sudah disimpan
    scaler = joblib.load('scaler.pkl')  # Memuat scaler yang sudah disimpan
    return model, scaler

# Load model and scaler
model, scaler = load_model_and_scaler()  # Memuat model dan scaler

# Main input form for new data point
st.header("Prediksi Kondisi Kesehatan Mental")

# Membuat form input data baru
with st.form(key='data_form'):
    heart_rate = st.number_input("Masukkan Heart Rate (BPM)", min_value=0.0, step=0.1)
    sleep_duration = st.number_input("Masukkan Sleep Duration (Hours)", min_value=0.0, step=0.1)
    physical_activity = st.number_input("Masukkan Physical Activity (Steps)", min_value=0, step=1)
    mood_rating = st.number_input("Masukkan Mood Rating (1-10)", min_value=1, max_value=10, step=1)

    # Submit button
    submit_button = st.form_submit_button(label='Prediksi')

# Proses data dan tampilkan hasil prediksi saat tombol submit ditekan
if submit_button:
    if heart_rate is not None and sleep_duration is not None and physical_activity is not None and mood_rating is not None:
        try:
            # Membuat list input data berdasarkan form yang dimasukkan
            new_data_point = [heart_rate, sleep_duration, physical_activity, mood_rating]
            
            # Melakukan scaling terhadap input data baru
            new_data_scaled = scaler.transform([new_data_point])

            # Prediksi menggunakan model
            gnb_new_pred = model.predict(new_data_scaled)

            # Menampilkan hasil prediksi dengan label deskriptif
            if gnb_new_pred[0] == 0:
                predicted_condition = "Tidak Ada Resiko"
                bg_color = "background-color: #d4edda; color: #155724;"  # Hijau terang
            else:
                predicted_condition = "Resiko Depresi Tinggi"
                bg_color = "background-color: #f8d7da; color: #721c24;"  # Merah terang
            
            # Menampilkan hasil prediksi dengan styling
            st.markdown(f'<div style="{bg_color}; padding: 10px; border-radius: 5px; text-align: center;">'
                        f'<strong>{predicted_condition}</strong></div>', unsafe_allow_html=True)
        
        except ValueError as e:
            # Menangani kesalahan jika input tidak valid
            st.write("Error: Input harus berupa angka yang valid.")
        except Exception as e:
            # Menangani kesalahan lain
            st.write("Terjadi kesalahan:", e)
    else:
        st.write("Mohon masukkan semua data sebelum menekan tombol prediksi.")
