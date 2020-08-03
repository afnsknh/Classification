from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import scipy
from PIL import Image
from random import randint
from naive_bayes import naive_bayes
import img_to_array
import img_to_array as convert

app = Flask(__name__) 
@app.route('/', methods = ['GET', 'POST'])
@app.route('/index',methods=['GET', 'POST'])
def main_page():
    proses = ""
##    print(proses)
    if request.method == 'POST' :

        nb = naive_bayes()
        dataset = nb.dataset
        proses = request.form['proses']
        print(proses, "proses")
        if proses == "training":
            input_test = (100 - int(request.form["id_train"])) / 100
            print("Persentase Data Test :", input_test)
            total_data = len(dataset)
            print("Jumlah Dataset :", total_data)
            data_test = int(total_data*input_test)
            print("Jumlah Data Test :", data_test)
            data_train = total_data-data_test
            print("Jumlah Data Train :", data_train)

            #random data test
            index_data = list(range(total_data))
            index_data_test = []
            while len(index_data_test) !=  data_test:
                ind_random = randint(0, len(dataset)-1)
                if ind_random not in index_data_test:
                    index_data_test.append(ind_random)

            index_data_train = np.delete(index_data, [index_data_test],0).tolist()
            new_data_train = np.delete(dataset, [index_data_test],0).tolist()

            #fit model
            model = nb.summarize_by_class(new_data_train)

            index_img = []
            hasil_prediksi = []
            benar = 0
            salah = 0
            for index in index_data_test:
                if index < 846:
                    index_img.append(img_to_array.file_kelas_0[index])
                elif index >= 846 and index < 1787:
                    index_1 = index - 846
                    index_img.append(img_to_array.file_kelas_1[index_1])
                elif index >= 1787 and index < 2009:
                    index_2 = index - 1787
                    index_img.append(img_to_array.file_kelas_2[index_2])
                elif index >= 2009 and index <= 2034:
                    index_3 = index - 2009
                    index_img.append(img_to_array.file_kelas_3[index_3])

                row = dataset[index][0:3]
                kelas_asli = dataset[index][2]
                label = nb.predict(model, row)

                label_prediksi = list(nb.dict_kelas.keys())[list(nb.dict_kelas.values()).index(label)]
                hasil_prediksi.append(label_prediksi)
                if label==kelas_asli:
                    benar+=1
                else:
                    salah+=1
            akurasi=(benar/(benar+salah)) * 100
            return render_template('index.html', proses=proses, jml_data_train=data_train, jml_data_test=data_test, index_data_train=index_data_train, index_data_test=index_data_test, benar=benar, salah=salah, akurasi=akurasi)

        else:
            file = request.files["input_test"]
            img = Image.open(file.stream)
##            print(img)
            req_path = file.filename
            test_path = 'static/assets/data_test/' + req_path
##            print(test_path)
##            img = cv2.imread(test_path)
            img.save(test_path)
            nms, non_nms = convert.get_corner_value(test_path)
            row = [nms, non_nms]

            model = nb.summarize_by_class(dataset)

            label = nb.predict(model, row)

            print('Data= ',row,' Predicted: ' , label,'/',list(nb.dict_kelas.keys())[list(nb.dict_kelas.values()).index(label)])

            hasil_label_prediksi = list(nb.dict_kelas.keys())[list(nb.dict_kelas.values()).index(label)]
            nilai_probabilitas = nb.calculate_class_probabilities(model, dataset[0])
            nilai_probabilitas_baru = {}
            print(nb.dict_kelas)
            for key in nb.dict_kelas:
                value_temp = nb.dict_kelas[key]
                nilai_probabilitas_baru[key] = float(nilai_probabilitas[value_temp])

            return render_template('index.html', proses=proses, hasil_prediksi=hasil_label_prediksi, nilai_corner=row, nilai_probabilitas=nilai_probabilitas_baru, path=req_path)

            
    else:
        return render_template('index.html', proses=proses)

if __name__ == "__main__":
    app.run()

