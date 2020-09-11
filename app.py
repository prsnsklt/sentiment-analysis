import streamlit as st
import altair as alt
import os
import pandas as pd
import re
import string
from tqdm import tqdm_notebook as tqdm  
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, validation_curve, cross_validate, KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
# import nltk
import plotly.express as px
import plotly.graph_objects as go
import itertools


image_url = ('gambarcovid.jpg')



def main():
    
    

    halaman = st.sidebar.selectbox("MENU", ["Tentang", "Data Preprocessing","Klasifikasi Sentimen","Hasil Klasifikasi"])

    if halaman == "Tentang":
        def loadpage():
            st.markdown('''
            <h1 style="margin-bottom:0px">Implementasi Algoritma Klasifikasi Support Vector Machine Untuk Sentimen Analisis Pengguna Twitter Terhadap Kebijakan PSBB</h1>
            ''',unsafe_allow_html=True)
            st.markdown('''
            <hr>
            <div>
                <h3 class="title">Abstrak</h3>
                <p class="abstrak" align="justify">Pada saat ini penggunaan Twitter semakin luas. Semua pengguna twitter yang dapat dengan bebas untuk berpendapat dan membagikan sudut pandang mereka mengenai isu tren dunia membuat konten twitter menjadi beragam dan menarik untuk dianalisa, termasuk dengan tren kebijakan pemrintah yang ramai diperbincangkan di Indonesia. Munculnya pandemi Covid-19 ini membuat  pemerintah mengeluarkan kebijakan yang bertujuan untuk menekan laju pertambahan orang yang terinfeksi virus. Kebijakan ini diberi nama Pembatasan Sosial Berskala Besar atau yang dikenal PSBB. Kebijakan ini pun hangat diperbincangkan di sosial media Twitter, hal ini menjadi alasan kuat untuk dilakukan analisa sentimen terhadap kebijakan tersebut. Analisa sentimen dilakukan dengan menggunakan Support Vector Machine sebagai algoritma melakukan klasifikasi pada data tweet sebanyak 6.979 data, serta dilakukan pelabelan data menggunakan metode Lexicon Based. Setelah dilakukan pengukuran performa model klasifikasi terhadap 4 kernel Support Vector Machine diantaranya Linear, RBF, Polynomial dan Sigmoid menggunakan k-fold cross validation diperoleh nilai performa model klasifikasi dengan kernel Linear merupakan yang terbaik secara keseluruhan diantara kernel lainnya dalam penelitian analisa sentimen ini. Nilai accuracy, precision, recall, dan f1-score-nya  yang diperoleh model klasifikasi dengan kernel Linear secara berturut-turut adalah 88.9%,  86,71%, 99,12%, dan 89,78%. Model klasifikasi dengan kernel linear yang dibuat berhasil memprediksi setiap partisi data uji dalam pengujian model klasifikasi dengan k-fold cross validation sebanyak 3.929 (56.3%) data tweet kedalam kelas positif dan 3.049 (43.7%) data tweet kedalam kelas negatif, sehingga dapat disimpulkan bahwa tweet pengguna twitter cenderung bersentimen positif terhadap kebijakan PSBB.</p>
            </div>               
            ''',unsafe_allow_html=True)
            if st.checkbox("Tentang Penulis dan Pembimbing"):
                st.markdown('''
                <div id='container'>
                    <div id="conten">
                        <h2 class="nama">Penulis</h2>
                        <p class="biodata">Nama: Heart Parasian PR Zuriel<br>Perguruan Tinggi : Universitas Gunadarma<br>jurusan : Teknik Informatika<br>NPM : 53416253<br>JK : Laki-laki<br>TTL : Medan, 17 Agustus 2020<br>Agama : Kristen<br></p>
                    </div>
                </div>
                <div>
                    <div id='parent'>
                        <div id='wide'>
                            <h2 class="title">Dosen Pembimbing </h2>
                            <p class="biodata1">Nama Lengkap : DR. Achmad Fahrurozi, S.Si, M.Si<br>NIDN : 0312058701</p>
                        </div>
                    </div>
                </div>              
            ''',unsafe_allow_html=True)
        loadpage() 
        
    elif halaman == "Data Preprocessing":
        st.markdown(
        """
        <h1 class="title">Data Preprocessing</h1>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <h2 class="title">Informasi Tentang Data</h2>
        <ul>
            <li>Data diperoleh dari Media Sosial Twitter dengan memanfaatkan Twitter API</li>
            <li>Kata kunci pencarian tweet adalah : 'psbb' </li>
            <li>Data dikumpulkan pada tanggal 28 Mei 2020</li>
            
        </ul>
        """, unsafe_allow_html=True)

        dataTweet = pd.read_csv('editedcovidlexpalingbaru.csv')
        dataTweet = dataTweet[['id','original_text']]
        st.markdown(
        """
        <h2 class="title">Data Awal</h2>
        """, unsafe_allow_html=True)
        
        if st.checkbox('Menampilkan Seluruh data',key=0):
            st.markdown(
                """
                <h3>Menampilkan Seluruh data</h3>
                """, unsafe_allow_html=True)
            fig = go.Figure(data=[go.Table(
                columnorder = [1,2],
                columnwidth = [100,500],
                header=dict(values=list(dataTweet),
                            fill_color='grey',
                            align='center',
                            height=30,
                            font=dict(color='white')),
                cells=dict(values=[dataTweet.id, dataTweet.original_text],
                        fill_color='white',
                        align='left',
                        height=50))
            ])
            fig.update_layout(width=850, height=400)
            st.plotly_chart(fig)
#             st.markdown(
#                 """
#                 <h3>Menampilkan Bentuk Data</h3>
#                 """, unsafe_allow_html=True) 
#             st.write(dataTweet.shape)
#             st.write("Jumlah baris ",dataTweet.shape[0] ," Jumlah kolom ", dataTweet.shape[1])
    

        ################################## CASE FOLDING #######################################    

        st.markdown(
        """
        <h2 class="title">Data Preprocessing | Case Folding</h2>
        """, unsafe_allow_html=True)
        def casefolding(text):
            text = text.lower().strip()
            return text
        hasilcasefolding = []
        for text in dataTweet.original_text:
            pro = casefolding(text)
            hasilcasefolding.append(pro)
            
        dataTweet.insert(1,"casefolding",hasilcasefolding)
        dataTweet.to_csv('Hasilcasefolding.csv')

        if st.checkbox('Hasil Preprocessing | Case Folding',key=1):
            fig = go.Figure(data=[go.Table(
                columnorder = [1,2],
                columnwidth = [400,400],
                header=dict(values=list(dataTweet),
                            fill_color='grey',
                            align='center',
                            height=30,
                            font=dict(color='white')),
                cells=dict(values=[dataTweet.original_text, dataTweet.casefolding],
                        fill_color='white',
                        align='left',
                        height=50))
            ])
            fig.update_layout(width=850, height=400)
            st.plotly_chart(fig)


        ################################## NOISE REMOVAL #######################################
        st.markdown(
        """
        <h2 class="title">Data Preprocessing | Cleaning</h2>
        """, unsafe_allow_html=True)
        def clean(text):
            text = re.sub('RT[\s]+', '', text)
            text = re.sub('rt[\s]+', '', text)
            text = re.sub(r'\n', '', text)
            #Menghapus username 
            text = re.sub("(@[A-Za-z0-9]+)","",text)
            text = re.sub("(@_[A-Za-z0-9]+)","",text)
            text =  re.sub(r'\\x..', '', text)
            #Menghapus #
            text = re.sub("(#[A-Za-z0-9]+)","",text)
            text = re.sub("(\w+:\/\/\S+)","",text)
            #Menghapus link
            text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', text)
            #Menghapus sebuah kalimat bersama angka dalam satu kata
            text = re.sub("(\w*\d\w*)","",text)
            # menghapus simbol
            text = re.sub(r'[^A-Za-z\s\/]' ,' ', text)
            text = re.sub(r'_', '', text) #hapus simbol _
            # menghapus angka
            text = re.sub(r'\d+', '', text)  
            # menghapus spasi
            text = re.sub(r'\s{2,}', ' ', text)
            text = re.sub('[^a-zA-Z]', ' ', text)
            #menghapus b diawal kalimat tweet
            # text = text.replace('b ','')

            text = re.sub('tdk','tidak', text)
            text = re.sub('nggak','tidak', text)

            text = re.sub('ndak','tidak', text)


            return text
  
        hasilclean = []
        for desc in dataTweet.casefolding:
            pro = clean(desc)
            hasilclean.append(pro)
  
        dataTweet.insert(2,"cleaning",hasilclean)
        dataTweet.to_csv('Hasilcleaning.csv')
        dataTweet.rename(columns={"text": "Hasil Cleaning"}).head(10)

        if st.checkbox('Hasil Preprocessing | Cleaning',key=2):
            fig = go.Figure(data=[go.Table(
                columnorder = [1,2,3],
                columnwidth = [400,400,400],
                header=dict(values=list(dataTweet),
                            fill_color='grey',
                            align='center',
                            height=30,
                            font=dict(color='white')),
                cells=dict(values=[dataTweet.original_text, dataTweet.casefolding, dataTweet.cleaning],
                        fill_color='white',
                        align='left',
                        height=50))
            ])
            fig.update_layout(width=850, height=400)
            st.plotly_chart(fig)

        
        ################################## Tokenizer #######################################
        st.markdown(
        """
        <h2 class="title">Data Preprocessing | Tokenizer </h2>
        """, unsafe_allow_html=True)
        def tokenizer(text):
            return text.split()

        hasiltokenizer = []
        for desc in tqdm(dataTweet['cleaning']):
            pro = tokenizer(desc)
            hasiltokenizer.append(pro)
            
        dataTweet.insert(3,"tokenizer",hasiltokenizer)
        dataTweet.to_csv('hasiltokonizer.csv')
        dataTweet.rename(columns={"text": "Hasil Tokenizer"}).head(10)

        if st.checkbox('Hasil Preprocessing | Tokenizer ',key=3):
            fig = go.Figure(data=[go.Table(
                columnorder = [1,2,3,4],
                columnwidth = [400,400,400,400],
                header=dict(values=list(dataTweet),
                            fill_color='grey',
                            align='center',
                            height=30,
                            font=dict(color='white')),
                cells=dict(values=[dataTweet.original_text, dataTweet.casefolding, dataTweet.cleaning, dataTweet.tokenizer],
                        fill_color='white',
                        align='left',
                        height=50))
            ])
            fig.update_layout(width=850, height=400)
            st.plotly_chart(fig)

        ################################## StopWord Removal #######################################
        st.markdown(
        """
        <h2 class="title">Data Preprocessing | Stopword Removal </h2>
        """, unsafe_allow_html=True)

        def hapus_stopword(desc):
            StopWords = "data_preprocess/combined_stop_words.txt"
            sw=open(StopWords,encoding='utf-8', mode='r');stop=sw.readlines();sw.close()
            stop=[kata.strip() for kata in stop];stop=set(stop)
            kata = [item for item in desc if item not in stop]
            return ' '.join(kata)
        hasilstopword = []

        for desc in tqdm(dataTweet['tokenizer']):
            pro = hapus_stopword(desc)
            hasilstopword.append(pro)
            
        dataTweet.insert(4,"clean_tweet",hasilstopword)
        dataTweet.to_csv('Hasilstopwordfile.csv')
        if st.checkbox('Hasil Preprocessing | Stopword Removal ',key=4):
            fig = go.Figure(data=[go.Table( 
                columnorder = [1,2,3,4,5],
                columnwidth = [400,400,400,400,400],
                header=dict(values=list(dataTweet),
                            fill_color='grey',
                            align='center',
                            height=30,
                            font=dict(color='white')),
                cells=dict(values=[dataTweet.original_text, dataTweet.casefolding, dataTweet.cleaning, dataTweet.tokenizer,dataTweet.clean_tweet],
                        fill_color='white',
                        align='left',
                        height=50))
            ])
            fig.update_layout(width=1000, height=400)
            st.plotly_chart(fig)


    elif halaman == "Klasifikasi Sentimen":
        st.markdown(
        """
        <h1 class="title">Klasifikasi Sentimen</h1>
        """, unsafe_allow_html=True)
        st.image(image_url)
        st.markdown(
            
        """
        <p> Klasifikasi sentimen dilakukan dengan menggunakan algoritma klasifikasi Support Vector Machine (SVM)</p>
        """, unsafe_allow_html=True)

        dataTweet = pd.read_csv('editedcovidlexpalingbaru.csv')
        dataTweet = dataTweet[['id','original_text','clean_tweet','sentiment']]

        X = dataTweet.clean_tweet
        y = dataTweet.sentiment
        vectorizer = CountVectorizer()
        X_train_counts = vectorizer.fit_transform(X.values)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        
        st.markdown(
        """
        <h3>Data</h3>
        """, unsafe_allow_html=True)

        fig = go.Figure(data=[go.Table(
            columnorder = [1,2,3,4],
            columnwidth = [80,200,200,80],
            header=dict(values=list(dataTweet),
                        fill_color='grey',
                        align='center',
                        height=30,
                        font=dict(color='white')),
            cells=dict(values=[dataTweet.id, dataTweet.original_text, dataTweet.clean_tweet, dataTweet.sentiment],
                    fill_color='white',
                    align='left',
                    height=50))
        ])
        fig.update_layout(width=800 , height=400)
        st.plotly_chart(fig)
        # st.dataframe(dataTweet.head(10))
        st.write(dataTweet.shape)
        st.write("""
            Data diberi label positif = 1 dan negatif = -1  
        """)

        if st.checkbox('Frekuensi sentimen'):
            sen_count = dataTweet['sentiment'].value_counts()
            sns.barplot(sen_count.index, sen_count.values, alpha=0.8)
            plt.title('Frekuensi Sentimen')
            plt.ylabel('Jumlah')
            plt.xlabel('Sentimen')
            st.pyplot()    

        st.subheader('WordCloud')        
        word_to_plot = dataTweet['clean_tweet']
        wordcloud = WordCloud(width = 800, height = 800, background_color = 'white', max_words = 8000
                , min_font_size = 20).generate(str(word_to_plot))
        fig = plt.figure(facecolor = None)
        plt.imshow(wordcloud)
        plt.axis('off')
        st.pyplot()
        # st.write('Data yang digunakan untuk training dan testing algoritma klasifikasi dibagi menjadi dua bagian dengan perbandingan 80:20, sehingga terdapat',len(X_train),'data latih dan terdapat',len(X_test),'data uji.')
        st.markdown(
        """
        <h3>Feature Extraction</h3>
        <p>Feature Extraction merupakan pembobotan data pada dokumen yang akan digunakan untuk melatih model menggunakan algoritma TF-IDF</p><br>
        
        """, unsafe_allow_html=True)
        
        if st.checkbox('Tampilkan occurence', key=0):
    
            occ = np.asarray(X_train_counts.sum(axis=0)).ravel().tolist()
            counts_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences': occ})
            st.dataframe(counts_df.sort_values(by='occurrences', ascending=False).head(20))


        if st.checkbox('Tampilkan weights', key=1):
        
            weights = np.asarray(X_train_tfidf.mean(axis=0)).ravel().tolist()
            weights_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'weight': weights})
            st.dataframe(weights_df.sort_values(by='weight', ascending=False).head(20))
        

        #====== Sidebar =======#
        # model = st.sidebar.selectbox("Algoritma Klasifikasi",["SVM"])
       

        #==================================Modelling=======================================#
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        params = dict()        
        kernel = st.sidebar.selectbox('Pilih Kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
        params['kernel'] = kernel
        clf = SVC(kernel=params['kernel'])
        scoring = {'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score),
                    'recall': make_scorer(recall_score), 
                    'f1': make_scorer(f1_score),}
        cv_result = cross_validate(clf, X_train_tfidf, y, scoring=scoring, cv=kfold) #proses pemisahan dataset, pelatihan model dan pelatihan model
        prediction = cross_val_predict(clf, X_train_tfidf, y, cv=kfold)
        conf_mat = confusion_matrix(y, prediction)
        st.header("Hasil Evaluasi")
        st.markdown("""
        <p>Setelah model klasifikasi di latih dan kemudian di uji dengan data uji berulang sebanyak 5 kali (k=5) menggunakan k-fold cross validation, berikut hasil rata-rata accuracy, precision, recall, dan fscore. </p>
        
        """, unsafe_allow_html=True) 
        
        st.write('Accuracy : ', round(cv_result['test_accuracy'].mean()*100, 2),'%')
        st.write('Precision : ',round(cv_result['test_precision'].mean()*100, 2),'%')
        st.write('Recall : ',round(cv_result['test_recall'].mean()*100, 2),'%')
        st.write('F1-score : ',round(cv_result['test_f1'].mean()*100, 2),'%')

        st.subheader('Tabel Confusion Matrix')
 
        def plot_confusion_matrix(cm,
                                target_names,
                                title='Confusion matrix',
                                cmap=None,
                                normalize=True):

            if cmap is None:
                cmap = plt.get_cmap('YlGnBu')

            plt.figure(figsize=(7, 5))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=90)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")


            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.grid(False)
            st.pyplot()        
       
        plot_confusion_matrix(cm = conf_mat, 
                                normalize    = False,
                                target_names = ['-1','1'],
                                title        = "Confusion Matrix SVC")


        # st.header("Hasil Validasi")
        # st.markdown("""
        # <p>Validasi dilakukan dengan menggunakan Cross Validation dengan jumlah k-fold = 10, menghasilkan nilai akurasi berikut.  </p>
        
        # """, unsafe_allow_html=True) 
        # scoring=['accuracy', 'precision', 'recall']
        # # cross_val = pd.DataFrame(cross_validate(clf, X_test_tfidf, y_test, scoring=scoring,cv=10))
        # cross_val = pd.DataFrame(cross_val_score(clf, X_test_tfidf, y_test,cv=5))

        # cross_val.rename(columns={0:'Accuracy'}, inplace=True)
        # st.dataframe(cross_val)
        # visualizer = CVScores(clf, cv=5 )
        # visualizer.fit(X_test_tfidf, y_test)
        
        # st.subheader('Bar Chart Cross Validation')
        # plt.ylabel('Accuracy')
        # plt.xlabel('K-fold')
        # st.pyplot()
                

    elif halaman == "Hasil Klasifikasi":
        st.markdown(
        """
        <h1 class="title">Hasil Klasifikasi</h1>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <p class="title">Setelah model klasifikasi dilatih dengan data latih, kemudian diuji kemampuan prediksinya dengan data uji. </p>
        
        """, unsafe_allow_html=True)
        
        #Feature Extraction
        dataTweet = pd.read_csv('editedcovidlexpalingbaru.csv')
        
        X = dataTweet.clean_tweet
        y = dataTweet.sentiment
        X_original = dataTweet.original_text

        vectorizer = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        X_train_counts = vectorizer.fit_transform(X)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        params = dict()        
        kernel = st.sidebar.selectbox('Pilih Kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
        params['kernel'] = kernel
        clf = SVC(kernel=params['kernel'])
        
        prediction = cross_val_predict(clf, X_train_tfidf, y, cv=kfold)

        positive = (prediction== 1).sum()
        negative = (prediction== -1).sum()
        st.write('Positif: ', positive, 'tweet')
        st.write('Negatif: ', negative, 'tweet')
        st.write('Dari', len(prediction), 'data')

        #penggabungan prediksi dan tweet
        X = pd.DataFrame(X)
        
        X_original = pd.DataFrame(X_original)
        df = pd.DataFrame(data=prediction, columns=["sentiment"])
        X_test_hasil = X.rename(columns={0:'clean_tweet'})
        X_test_hasil = X_test_hasil.join(df)
        X_test_original = X_original.reset_index()

        del X_test_original['index']
        X_test_original = X_test_original.join(X_test_hasil)
        del X_test_original['clean_tweet']

        #tweet positif
        st.subheader('Tabel Tweet Positif')
        X_test_positif = X_test_original.loc[X_test_original['sentiment']==1]
        X_test_positif = X_test_positif.reset_index()
        del X_test_positif['index']
        # st.table(X_test_positif)

        fig = go.Figure(data=[go.Table(
            columnorder = [1,2],
            columnwidth = [500,80],
            header=dict(values=list(X_test_positif),
                        fill_color='grey',
                        align='center',
                        height=30,
                        font=dict(color='white')),
            cells=dict(values=[X_test_positif.original_text, X_test_positif.sentiment],
                    fill_color='white',
                    align='left',
                    height=50))
        ])
        fig.update_layout(width=850, height=400)
        st.plotly_chart(fig)
        
        # #tweet negatif
        st.subheader('Tabel Tweet Negatif')
        X_test_negatif = X_test_original.loc[X_test_original['sentiment']==-1]
        X_test_negatif = X_test_negatif.reset_index()
        del X_test_negatif['index']
        # st.dataframe(X_test_negatif)
        fig = go.Figure(data=[go.Table(
            columnorder = [1,2],
            columnwidth = [500,80],
            header=dict(values=list(X_test_negatif),
                        fill_color='grey',
                        align='center',
                        height=30,
                        font=dict(color='white')),
            cells=dict(values=[X_test_negatif.original_text, X_test_negatif.sentiment],
                    fill_color='white',
                    align='left',
                    height=50))
        ])
        fig.update_layout(width=850, height=400)
        st.plotly_chart(fig)

        #pie chart
        st.subheader('Pie Chart Sentimen')
        pie_data = df['sentiment'].value_counts()
        def func(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return "{:.1f}%)".format(pct, absolute)
        
        fig1, ax1 = plt.subplots()
        ax1.pie(pie_data, autopct=lambda pct: func(pct, pie_data), textprops=dict(color="w"))
        ax1.set_title("Presentase Klasifikasi Sentimen")
        ax1.axis('equal')
        label = {'1','-1'}
        ax1.legend(label)
        st.pyplot()
            
                    
          
                  

                
if __name__ == "__main__":
    main()
