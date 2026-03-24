# 1. Temporary container
docker run --rm --gpus all -it \
-v /home/alex/projects/Dissertation-Workspace:/workspace \
-v "/mnt/c/Users/Alex/OneDrive/Desktop/Year 3/0 - Dissertation/Datasets/Google Speech Commands:/data" \
-w /workspace \
tensorflow/tensorflow:2.16.1-gpu \
bash

# 2. Reinstall packages
pip install matplotlib pandas scikit-learn
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"