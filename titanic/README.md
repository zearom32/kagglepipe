# titanic
For titanic competition: https://www.kaggle.com/c/titanic

Requirements:

-   tensorflow>=2.0
-   tfx
-   docker
-   tensorflow serving

```shell
pip install --user kaggle
kaggle competitions download -c titanic
mkdir -p /tmp/titanic/data/train
mkdir /tmp/titanic/data/test
unzip titanic.zip -d /tmp/titanic/data
# copy this folder to /tmp/titanic
cp -r <path_to>/kagglekfp/titanic/* /tmp/titanic/
cd /tmp/titanic/

# Generate training TfRecord
python3 data_pipeline.py --input_file=/tmp/titanic/data/train.csv --output_file=/tmp/titanic/data/train/train.tfrecord
# Generate testing TfRecord
python3 data_pipeline.py --input_file=/tmp/titanic/data/test.csv --output_file=/tmp/titanic/data/test/test.tfrecord

# Train model and export saved model.
python3 titanic_keras.py --project_root=/tmp/titanic/ --data_root=/tmp/titanic/data/train/

# Run serving
docker run -p 8500:8500 --mount type=bind,source=\
/tmp/titanic/serving_model/titanic/,target=/models/titanic \
-e MODEL_NAME=titanic -t tensorflow/serving
# Get submission.csv
python3 client_keras.py --test_data=/tmp/titanic/data/test/test.tfrecord --output_file=/tmp/titanic/submission.csv

# Cleanup
cd ~
rm -rf /tmp/titanic/
```
