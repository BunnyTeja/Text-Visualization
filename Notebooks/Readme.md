Improving Domain Classification Of KITE using Heirarchical Transformer Model

NOTE : BEFORE RUNNING THIS CODE, PLEASE FOLLOW THE INSTRUCTIONS OF THE README PROVIDED IN THE 'Text-Visualization' parent folder.

---DESCRIPTION---

1. data : Training data for the model
2. model_weights : Consists of different weights stored after each epoch of training, can be loaded for experimentation and testing.
3. Results : Contains .csv files of model predictions for all available URls which have been run on the system.
4. training_photos : Some snapshots taken during training of the model.

--- RUNNING THE CODE ---
1. Download all requirements as mentioned in "requirements.txt" in Text-Visualization (pip install -r requirements.txt)
2. Run v4-LSTM-Testing (py v4-LSTM-Testing.py)

--- EXPERIMENTATION WITH THE CODE ---
1. 'model_weights' contains different versions of heirarchical transformer model. Each can be loaded as : model.load_weights(r'./model_weights/bert-lstm/v7.h5')  [ line 139 in v4-LSTM-Testing.py]

--- JUPYTER NOTEBOOKS ---
1. v4-LSTM : Notebook used for creating the training data and model.
2. v4-LSTM-Testing : Notebook for only testing the Hierarchical transformer model
