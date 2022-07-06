# plds_teamc: Live-birth occurence prediction
Proyek ini bertujuan untuk mengetahui kemungkinan keberhasilan kehamilan berdasarkan data yang di-input pasien IVF.
## Sumber data dan model

Data diambil dari Human Fertilisation & Embryology Authority (https ://www.hfea.gov.uk) dari tahun 2010-2016. Dari 90++ fitur yang ada, dipilih fitur yang sama sesuai dengan paper acuan (Goyal, A., Kuchana, M. & Ayyagari, K.P.R. Machine learning predicts live-birth occurrence before in-vitro fertilization treatment. Sci Rep 10, 20925 (2020). https://doi.org/10.1038/s41598-020-76928-z). 

Model yang digunakan adalah deep learrning model, dengan arsitektur yang sama dengan paper tersebut.

## Cara menggunakan

Untu saat ini, hanya bisa dijalankan melalui notebook ()google colab/jupyter) saja. File utama adalah src/main.ipynb.

Setelah file yang berisi data pasien diisi (data/txtfile.csv), pada notebook terbeut arahkan path dari model, threhold, dan data pasien sesuai dengan relatif path-nya masing-masing.
Setelah semua cell dijalankan, akan muncul hasil kemungkinan terjadi kehamilan, disertai dengan akurasinya.


