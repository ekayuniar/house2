import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from fcmeans import FCM

st.write("""
# Clustering Fuzzy C Means
""")

fcmm3 = pickle.load(open('./model/model_fcm2.pkl', 'rb'))


def run():
    o = {1: 'Ada', 0: 'Tidak Ada'}
    olga = list(o.keys())
    fas_olga = st.selectbox(
        'Fasilitas Olahraga', olga, format_func=lambda x: o[x])

    aman = {1: 'Ada', 0: 'Tidak Ada'}
    fas_aman = list(aman.keys())
    fas_keamanan = st.selectbox(
        'Fasilitas Keamanan', fas_aman, format_func=lambda x: aman[x])

    jrk_airport = st.number_input(
        'Input Jarak Airport terdekat dalam satuan Km')
    jrk_toll = st.number_input(
        'Input Jarak Toll terdekat dalam satuan Km')
    jrk_transport = st.number_input(
        'Input Jarak Transportasi umum terdekat dalam satuan Km')
    jrk_supermarket = st.number_input(
        'Input Jarak supermarket terdekat dalam satuan Km')

    hm = {1: 'Modern', 2: 'Minimalis', 3: 'Klasik Modern', 4: 'Kontemporer'}
    house_m = list(hm.keys())
    house_model = st.selectbox(
        'Model Rumah', house_m, format_func=lambda x: hm[x])

    luas_rumah = st.number_input('Input Luas Tanah dalam M2')
    uk_rumah = st.number_input('Input Ukuran Bangunan Rumah dalam M2')

    hrg_beli = st.number_input('Input Harga Beli')
    st.subheader('Tabel Inputan Data')
    data = {'Fasilitas Olahraga': fas_olga,
            'Fasilitas Keamanan': fas_keamanan,
            'Jarak Airport': jrk_airport,
            'Jarak Toll': jrk_toll,
            'Jarak Transport': jrk_transport,
            'Jarak Supermarket': jrk_supermarket,
            'Model Rumah': house_model,
            'Luas Rumah': int(luas_rumah),
            'Ukuran Rumah': int(uk_rumah),
            'Harga Beli': int(hrg_beli)}
    fitur = pd.DataFrame(data, index=[0])
    st.write(fitur)

    st.subheader('Hasil Clustering Rumah dengan Fuzzy C Means')
    if st.button("Submit"):
        fitur = pd.DataFrame(data, index=[0])
        print(fitur)
        prediction = fcmm3.predict(fitur.values)
        lc = [str(i) for i in prediction]
        ans = int("".join(lc))
        keterangan = np.array(0)
        labels2 = (prediction[keterangan])
        if ans == 0:
            st.error(
                labels2
            )
        else:
            st.success(
                labels2
            )


run()
