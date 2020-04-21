# Santander

https://www.kaggle.com/c/santander-customer-transaction-prediction

# Prepare enviroment

1. Install latest TF, TFX

To get a pretty clean enviroment, please use virtualenv

```
virtualenv -p python3 ~/envtfx
source ~/envtfx/bin/activate
pip install tensorflow
pip install tfx
pip install kaggle
```

2. Download data

```
mkdir -p /var/tmp/santander/data/train
mkdir -p /var/tmp/santander/data/test
cd /var/tmp/santander/data
kaggle competitions download -c santander-customer-transaction-prediction
unzip santander-customer-transaction-prediction.zip
rm santander-customer-transaction-prediction.zip
mv train.csv train/
mv test.csv test/
cd <this_folder>
```

# Estimator-based

```
cd estimator
python tfxbeam.py
```

# Keras without TFT

```
cd keras
python tfxbream.py
```

# Keras with TFT (preferred)

```
cd keras-tft
python tfxbeam.py
```

                                                                                
