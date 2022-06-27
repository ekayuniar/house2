import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from fcmeans import FCM

st.write("""
# Clustering Fuzzy C Means
""")

fcm = pickle.load(open('./model/model_fcm2.pkl', 'rb'))


def run():
    o = {1: 'Ada', 0: 'Tidak Ada'}
    olga = list(o.keys())
    fas_olga = st.selectbox(
        'Fasilitas Olahraga', olga, format_func=lambda x: o[x])

    aman = {1: 'Ada', 0: 'Tidak Ada'}
    fas_aman = list(aman.keys())
    fas_keamanan = st.selectbox(
        'Fasilitas Keamanan', fas_aman, format_func=lambda x: aman[x])

    airport = {1: '< 5 km', 2: '5 - 10 km', 3: '20 - 40 km',
               4: '40 - 60 km', 5: '60 - 100 km', 6: '100 - 200 km', 7: '>200 km'}
    jrk_airport1 = list(airport.keys())
    jrk_airport = st.selectbox(
        'Jarak Airport', jrk_airport1, format_func=lambda x: airport[x])

    t = {1: '< 5 km', 2: '5 - 10 km', 3: '20 - 30 km',
         4: '30 - 40 km', 5: '40 - 50 km', 6: '100 - 200 km', 7: '>200 km'}
    toll = list(t.keys())
    jrk_toll = st.selectbox('Jarak Toll', toll, format_func=lambda x: t[x])

    jrk_transport = st.number_input(
        'Jarak Transportasi', min_value=None, max_value=None)
    jrk_supermarket = st.number_input(
        'Jarak Supermarket', min_value=None, max_value=None)

    hm = {1: 'Modern', 2: 'Minimalis', 3: 'Klasik Modern', 4: 'Kontemporer'}
    house_m = list(hm.keys())
    house_model = st.selectbox(
        'Model Rumah', house_m, format_func=lambda x: hm[x])

    luas_rumah = st.number_input(
        'Luas Rumah dalam satuan m2', min_value=1, max_value=None)
    uk_rumah = st.number_input(
        'Ukuran Rumah dalam satuan m2', min_value=None, max_value=None)
    hrg_beli = st.number_input(
        'Silahkan input Harga Rumah dalam satuan Ratusan Juta/Milyar', step=1, min_value=1, max_value=None)

    st.subheader('Tabel Inputan Data')
    data = {'Fasilitas Olahraga': fas_olga,
            'Fasilitas Keamanan': fas_keamanan,
            'Jarak Airport': jrk_airport,
            'Jarak Toll': jrk_toll,
            'Jarak Transport': jrk_transport,
            'Jarak Supermarket': jrk_supermarket,
            'Model Rumah': house_model,
            'Luas Rumah': luas_rumah,
            'Ukuran Rumah': uk_rumah,
            'Harga Beli': hrg_beli}
    fitur = pd.DataFrame(data, index=[0])
    st.write(fitur)
# print(arr[int(my_float)])
    prediksi = fcm.centers[fitur]
    labels = fcm.u.argmax(axis=1)

    #st.subheader('Keterangan Label Kelas')
    #keterangan = np.array(['MEWAH', 'TIDAK MEWAH'])
    # st.write(keterangan)

    st.subheader('Hasil Clustering Rumah dengan Fuzzy C Means')
    keterangan = np.array(0)
    st.write(labels[keterangan])


run()
