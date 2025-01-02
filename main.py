import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import time
st.set_page_config(page_title='The Machine Learning App',
    layout='wide')
st.write("## Clustering Web Applications")

data_format = {
    "Column": ["COMPOUND", "YEAR", "STATE_CODE", "COUNTY_CODE",
               "LOW_ESTIMATE", "HIGH_ESTIMATE", "COUNTY", "STATE"],
    "Description": ["Nama Pestisida", "Tahun Pengukuran", "Kode Negara", "Kode Wilayah",
                    "Metode estimasi rendah dalam kilogram",
                    "Metode estimasi tinggi dalam kilogram", "Nama Negara", "Nama Wilayah"]
}

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Processing", "Evaluate", "Output"])

with tab1:
    st.write("""
    This web application aims to cluster the dataset on pesticide use in agriculture in the format below.

    Please upload the file using the format below
    """)
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**Dataset Pesticide Use in Agriculture**')
        st.write(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        st.table(data_format)
        st.session_state.clear()

with tab2:
    st.write("## Processing")

    # Cek apakah file telah diunggah
    if uploaded_file is not None:

        if 'matrix' in st.session_state:
            st.success("Data setelah Cleaning")
            col1, col2 = st.columns(2, border=True, gap="medium", vertical_alignment="top", )
            col3, col4 = st.columns([1, 1], border=True)
            df.fillna({'LOW_ESTIMATE': 0}, inplace=True)

            # Tampilkan visualisasi yang telah disimpan
            st.write("### Visualisasi (Boxplot) Data setelah Cleaning")
            col1.write("#### Nilai Null")
            col1.table(df.isnull().sum())
            col3.pyplot(st.session_state.histogram)
            col2.pyplot(st.session_state.matrix)
            col4.pyplot(st.session_state.box)

        else:

            # Tombol untuk visualisasi
            if st.button("Cleaning Data"):

                st.success("Data telah selesai di cleaning")

                col1, col2 = st.columns(2, border=True, gap="medium", vertical_alignment="top",)
                col3, col4 = st.columns([1,1], border=True)

                df.fillna({'LOW_ESTIMATE': 0}, inplace=True)

                st.session_state.original_df = df.copy()

                col1.write("#### Nilai Null")
                col1.table(df.isnull().sum())

                # Histogram untuk LOW_ESTIMATE dan HIGH_ESTIMATE
                numeric_cols = ['LOW_ESTIMATE', 'HIGH_ESTIMATE']

                # Plot histogram
                col3.write("### Histogram of LOW_ESTIMATE and HIGH_ESTIMATE")
                fig, ax = plt.subplots(figsize=(12, 6))
                df[numeric_cols].hist(bins=20, edgecolor='black', ax=ax)
                plt.suptitle('Distribusi Nilai LOW_ESTIMATE dan HIGH_ESTIMATE')
                st.session_state.histogram = fig
                col3.pyplot(fig)

                # Missing value matrix
                col2.write("### Missing Values Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                msno.matrix(df, ax=ax, color=(0.27, 0.52, 1.0))
                st.session_state.matrix = fig
                col2.pyplot(fig)

                df = df.drop(['COMPOUND', 'COUNTY', 'STATE'], axis=1, errors='ignore')

                # Menggunakan IQR untuk menghapus outlier
                Q1h = df['HIGH_ESTIMATE'].quantile(0.25)
                Q3h = df['HIGH_ESTIMATE'].quantile(0.75)
                IQRh = Q3h - Q1h
                lower_boundh = Q1h - 1.5 * IQRh
                upper_boundh = Q3h + 1.5 * IQRh

                # Menggunakan IQR untuk menghapus outlier
                Q1l = df['LOW_ESTIMATE'].quantile(0.25)
                Q3l = df['LOW_ESTIMATE'].quantile(0.75)
                IQRl = Q3l - Q1l
                lower_boundl = Q1l - 1.5 * IQRl
                upper_boundl = Q3l + 1.5 * IQRl

                # Mengganti outlier dengan nilai median
                df['LOW_ESTIMATE'] = df['LOW_ESTIMATE'].apply(
                    lambda x: df['LOW_ESTIMATE'].mean() if x > upper_boundl or x < lower_boundl else x
                )

                # Mengganti outlier dengan nilai median
                df['HIGH_ESTIMATE'] = df['HIGH_ESTIMATE'].apply(
                    lambda x: df['HIGH_ESTIMATE'].mean() if x > upper_boundh or x < lower_boundh else x
                )

                df['RANGE'] = df['HIGH_ESTIMATE'] - df['LOW_ESTIMATE']

                st.session_state.df_cleaned = df

                # Boxplot untuk mendeteksi outlier
                col4.write("### Boxplot untuk mendeteksi outlier")
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(data=df[numeric_cols], ax=ax)
                ax.set_title('Boxplot untuk LOW_ESTIMATE dan HIGH_ESTIMATE')
                st.session_state.box = fig
                col4.pyplot(fig)

            else:
                col1, col2 = st.columns(2, border=True, gap="medium", vertical_alignment="top",)
                col3, col4 = st.columns([1,1], border=True)

                # Histogram untuk LOW_ESTIMATE dan HIGH_ESTIMATE
                numeric_cols = ['LOW_ESTIMATE', 'HIGH_ESTIMATE']

                col1.write("#### Nilai Null")
                col1.table(df.isnull().sum())

                # Plot histogram
                col3.write("#### Histogram of LOW_ESTIMATE and HIGH_ESTIMATE")
                fig, ax = plt.subplots(figsize=(12, 6))
                df[numeric_cols].hist(bins=20, edgecolor='black', ax=ax)
                plt.suptitle('Distribusi Nilai LOW_ESTIMATE dan HIGH_ESTIMATE')
                col3.pyplot(fig)

                # Missing value matrix
                col2.write("#### Missing Values Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                msno.matrix(df, ax=ax, color=(0.27, 0.52, 1.0))
                col2.pyplot(fig)

                # Boxplot untuk mendeteksi outlier
                col4.write("#### Boxplot untuk mendeteksi outlier")
                # Atur ukuran figure menggunakan figsize
                fig, ax = plt.subplots(figsize=(15, 8))  # Lebar 15 inci, tinggi 8 inci
                sns.boxplot(data=df[numeric_cols], ax=ax)
                ax.set_title('Boxplot untuk LOW_ESTIMATE dan HIGH_ESTIMATE', fontsize=12)  # Atur ukuran font judul
                ax.tick_params(axis='x', labelsize=12)  # Atur ukuran font untuk label sumbu X
                ax.tick_params(axis='y', labelsize=12)  # Atur ukuran font untuk label sumbu Y

                # Tampilkan grafik di kolom dengan ukuran penuh
                col4.pyplot(fig, use_container_width=True)




    else:
        st.warning("Please upload a CSV file to proceed.")

with tab3:
    st.write("## Modeling Clustering")

    # Cek apakah file telah diunggah dan di-cleaning
    if 'df_cleaned' in st.session_state:
        df = st.session_state.df_cleaned  # Ambil df yang sudah dibersihkan dari session state

        # Cek apakah data sudah memiliki kolom yang diperlukan
        if 'RANGE' in df.columns:
            # 4. Penerapan K-Means Clustering dengan 4 cluster
            kmeans = KMeans(n_clusters=4, random_state=0)
            df['Cluster'] = kmeans.fit_predict(df[['LOW_ESTIMATE', 'HIGH_ESTIMATE', 'RANGE']])


            # Menambahkan label kategori berdasarkan kondisi
            def label_cluster(row):
                if row['Cluster'] == 0:  # Jika LOW_ESTIMATE > HIGH_ESTIMATE
                    return 'Penggunaan Rendah'
                elif row['Cluster'] == 1:
                    return 'Penggunaan Tinggi'
                elif row['Cluster'] == 2:
                    return 'Penggunaan Berlebih/Anomali'
                else:
                    return 'Potensial Penggunaan'


            df['Cluster_Label'] = df.apply(label_cluster, axis=1)
            # Menyimpan hasil clustering ke session state
            st.session_state.df_clustered = df
            st.session_state.kmeans_model = kmeans

            col5 = st.columns([2, 1])[0]

            # 6. Visualisasi Hasil Clustering
            col5.write("### Scatter Plot Clustering")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='LOW_ESTIMATE', y='HIGH_ESTIMATE', hue='Cluster_Label', palette='viridis', ax=ax)
            ax.set_title('Clustering Penggunaan Pestisida Berdasarkan LOW_ESTIMATE dan HIGH_ESTIMATE')
            ax.set_xlabel('LOW_ESTIMATE')
            ax.set_ylabel('HIGH_ESTIMATE')
            col5.pyplot(fig)

            # 7. Menampilkan hasil klasterisasi dengan deskripsi kategori
            st.write("### Deskripsi Kategori Penggunaan")
            cluster_desc = df[['LOW_ESTIMATE', 'HIGH_ESTIMATE','Cluster', 'Cluster_Label']].groupby('Cluster_Label').mean()
            st.table(cluster_desc)

            # Evaluasi K-Means Clustering
            # Evaluasi K-Means Clustering
            st.write("### Evaluasi Clustering")
            st.write("Press the button to calculate")

            # Periksa apakah hasil evaluasi sudah ada di session state
            if 'silhouette_score' in st.session_state and 'dbi_score' in st.session_state:
                st.success("Hasil evaluasi sudah tersedia. Berikut hasilnya:")
                st.info(f"**Silhouette Score:** {st.session_state.silhouette_score:.2f}")
                st.info(f"**Davies-Bouldin Index:** {st.session_state.dbi_score:.2f}")
            else:
                # Tampilkan tombol untuk menghitung jika hasil belum ada
                if st.button("Calculate Silhouette Score and Davies-Bouldin Index"):
                    with st.spinner('Calculating Silhouette Score and Davies-Bouldin Index...'):
                        time.sleep(5)  # Simulasi waktu tunggu untuk menghitung metrik

                        # Lakukan perhitungan
                        silhouette_avg = silhouette_score(df[['LOW_ESTIMATE', 'HIGH_ESTIMATE']], df['Cluster'])
                        dbi_score = davies_bouldin_score(df[['LOW_ESTIMATE', 'HIGH_ESTIMATE']], df['Cluster'])

                        # Simpan hasil ke session state
                        st.session_state.silhouette_score = silhouette_avg
                        st.session_state.dbi_score = dbi_score

                    # Tampilkan hasil evaluasi
                    st.info(f"**Silhouette Score:** {silhouette_avg:.2f}")
                    st.info(f"**Davies-Bouldin Index:** {dbi_score:.2f}")



        else:
            st.warning("Data belum di-cleaning dengan benar. Silakan lakukan proses cleaning pada Tab 2.")
    else:
        st.warning("Pastikan data telah di-cleaning di Tab 2 terlebih dahulu.")

with tab4:
    st.write("## Output - Hasil Clustering")



    if 'df_clustered' in st.session_state and 'kmeans_model' in st.session_state and 'original_df' in st.session_state:

        df = st.session_state.df_clustered  # Ambil df yang sudah diproses dari session state
        kmeans = st.session_state.kmeans_model
        original_df = st.session_state.original_df

        if 'COMPOUND' in original_df.columns:
            df = pd.concat([df, original_df[['COMPOUND', 'COUNTY', 'STATE']]], axis=1)

        df['Cluster'] = kmeans.predict(df[['LOW_ESTIMATE', 'HIGH_ESTIMATE', 'RANGE']])

        # Pastikan df memiliki kolom yang dibutuhkan
        if 'Cluster' in df.columns:
            st.write("### Hasil Clustering")

            # Filter berdasarkan COUNTY
            filtered_df = df

            # Filter berdasarkan STATE
            state_options = filtered_df['STATE'].unique() if 'STATE' in filtered_df.columns else []
            selected_state = st.selectbox('Pilih STATE untuk filter', state_options)

            if selected_state:
                filtered_df = filtered_df[filtered_df['STATE'] == selected_state]

            # Menampilkan hasil filter
            st.write("### Data setelah filter")
            st.write(filtered_df)
        else:
            st.warning("Pastikan data telah di-cluster dan memiliki kolom 'Cluster_Label' di Tab 3.")
    else:
        st.warning("Pastikan data telah di-upload dan di-cleaning di Tab 2 dan Tab 3.")
