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
pip install tensorflow-model-analysis
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

3. Create pusher folder

Pusher can't create folder for you.

```
mkdir -p /var/tmp/santander/pusher
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

# Keras with TFT

## Run locally via Beam local runner

```
cd keras-tft
python tfxbeam.py --runner local
```

## Run in Kubeflow Pipelines

Install KFP & Skaffold
```
pip install kfp

# Install Skaffold
# https://skaffold.dev/docs/install/
```

```
cd keras-tft
tfx pipeline create --engine kubeflow \
    --build_target_image gcr.io/renming-mlpipeline/santander-custom \
    --pipeline_path tfxbeam.py \
    --endpoint 1c4774e7d442cc4c-dot-us-central2.pipelines.googleusercontent.com
```

It may takes a while in the phase of using skaffold to compile container image.
It will compile and upgrade to KFP.
