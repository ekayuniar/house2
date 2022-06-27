import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from fcmeans import FCM

st.write("""
# Clustering Fuzzy C Means
""")

fcm = pickle.load(open('./Model/model_fcm.pkl', 'rb'))


def run():
    o = {1: 'Ada', 0: 'Tidak Ada'}
    olga = list(o.keys())
    fas_olga = st.selectbox(
        'Fasilitas Olahraga', olga, format_func=lambda x: o[x])

    aman = {1: 'Ada', 0: 'Tidak Ada'}
    fas_aman = list(aman.keys())
    fas_keamanan = st.selectbox(
        'Fasilitas Keamanan', fas_aman, format_func=lambda x: aman[x])

    airport = {1: 'Dekat', 2: 'Jauh'}
    jrk_airport1 = list(airport.keys())
    jrk_airport = st.selectbox(
        'Jarak Airport', jrk_airport1, format_func=lambda x: airport[x])

    t = {1: 'Dekat', 2: 'Jauh'}
    toll = list(t.keys())
    jrk_toll = st.selectbox('Jarak Toll', toll, format_func=lambda x: t[x])

    trans = {1: 'Dekat', 2: 'Jauh'}
    transport = list(trans.keys())
    jrk_transport = st.selectbox(
        'Jarak Transport', transport, format_func=lambda x: trans[x])

    s = {1: 'Dekat', 2: 'Jauh'}
    smart = list(s.keys())
    jrk_supermarket = st.selectbox(
        'Jarak Supermarket', smart, format_func=lambda x: s[x])

    hm = {1: '1', 2: '2', 3: '3', 4: '4'}
    house_m = list(hm.keys())
    house_model = st.selectbox(
        'Model Rumah', house_m, format_func=lambda x: hm[x])

    lm = {1: 'Kecil', 2: 'Sedang', 3: 'Besar'}
    luas = list(lm.keys())
    luas_rumah = st.selectbox('Luas Rumah', luas, format_func=lambda x: lm[x])

    uk = {1: 'Kecil', 2: 'Sedang', 3: 'Besar'}
    ukuran = list(uk.keys())
    uk_rumah = st.selectbox('Ukuran Rumah', ukuran,
                            format_func=lambda x: uk[x])

    bl = {1: '1', 2: '2', 3: '2'}
    hrgb = list(bl.keys())
    hrg_beli = st.selectbox('Harga Beli', hrgb, format_func=lambda x: bl[x])

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

    prediksi = fcm.centers[fitur]
    labels = fcm.u.argmax(axis=1)

    #st.subheader('Keterangan Label Kelas')
    #keterangan = np.array(['MEWAH', 'TIDAK MEWAH'])
    # st.write(keterangan)

    st.subheader('Hasil Clustering Rumah dengan Fuzzy C Means')
    keterangan = np.array(0)
    st.write(labels[keterangan])


run()
