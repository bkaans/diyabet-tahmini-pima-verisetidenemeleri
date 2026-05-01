# EDA Raporu

## Veri Butunlugu
- Dosya: `makine_ogrenmesi/veri/ham/diabetes.csv`
- SHA256: `3c7023e0935097ac63e03c6f398928b6dfce30974d5591d3af09a9661876d9fe`
- Satir/Kolon: 768 / 9
- Sinif dagilimi: {0: 500, 1: 268}

## Sinif Oranlari
- Outcome=0: 0.651
- Outcome=1: 0.349

## 0 Deger Analizi
- Pregnancies: 111 adet (0.145)
- Glucose: 5 adet (0.007)
- BloodPressure: 35 adet (0.046)
- SkinThickness: 227 adet (0.296)
- Insulin: 374 adet (0.487)
- BMI: 11 adet (0.014)
- DiabetesPedigreeFunction: 0 adet (0.000)
- Age: 0 adet (0.000)

## Outcome Korelasyonu
- Glucose: 0.4666
- BMI: 0.2927
- Age: 0.2384
- Pregnancies: 0.2219
- DiabetesPedigreeFunction: 0.1738
- Insulin: 0.1305
- SkinThickness: 0.0748
- BloodPressure: 0.0651

## Sinif Bazli Ortalama Farklari
- Pregnancies: Outcome=0 3.298, Outcome=1 4.866, fark 1.568
- Glucose: Outcome=0 109.980, Outcome=1 141.257, fark 31.277
- BloodPressure: Outcome=0 68.184, Outcome=1 70.825, fark 2.641
- SkinThickness: Outcome=0 19.664, Outcome=1 22.164, fark 2.500
- Insulin: Outcome=0 68.792, Outcome=1 100.336, fark 31.544
- BMI: Outcome=0 30.304, Outcome=1 35.143, fark 4.838
- DiabetesPedigreeFunction: Outcome=0 0.430, Outcome=1 0.550, fark 0.121
- Age: Outcome=0 31.190, Outcome=1 37.067, fark 5.877

## Temel Feature Importance
- Glucose_BMI: 0.1349
- Glucose: 0.1312
- Glucose_Age: 0.1180
- BMI_Age: 0.0897
- BMI: 0.0765
- Age: 0.0728
- DiabetesPedigreeFunction: 0.0661
- Pregnancies: 0.0579
- Pregnancies_Age_Ratio: 0.0578
- BloodPressure: 0.0542
- SkinThickness: 0.0509
- Insulin: 0.0483
- Insulin_Glucose_Ratio: 0.0418

## Modelleme Notu
- Bu rapor ham CSV'yi degistirmez.
- 0 degerleri ve turetilmis ozellikler sadece model pipeline icinde denenmelidir.
