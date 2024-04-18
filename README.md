# FDU_CV_Assignment1
For all subsequent operations, it's important to maintain the following file structure as indicated in doc_tree.txt, as the location of the dataset folder is crucial for the file-reading processes within `main.py`:

### Testing
To test the model, download the model weights from Google Drive:
[Model Weights](https://drive.google.com/drive/folders/1uTwq6ZPBVR2kLWwqin_I10n2ULNaI40I?usp=sharing).

After downloading the weights to your local folder, you can test the model's performance on the test set with the following command: 
```python main.py --test-only --model-path model_parameters```


### Training
Set parameters in `config.py` following the comments provided within the file. To train the model, run: 
```python main.py```
This command will train the model and automatically save the parameters.

### Hyperparameter Searching
Use the following command for hyperparameter searching, allowing you to set the decay style and number of training epochs: 
```python hyperparameter.py --decay_style DS --epochs E```
The default search parameters include learning rate, batch size, hidden layer sizes, regularization strength, and learning rate decay coefficient. For additional adjustments, modifications can be made directly to `hyperparameter.py`.




