# FDU_CV_Assignment1
For all later operation, the file structure is like, the position of dataset folder matter to the following reading process in main.py:

D:\UNIVERSITY_RESOURCES\2023-2024 TERM_2\COMPUTER VISION\ASSIGNMENT_1
├─best_param
│  ├─epoches_100_lr_0.1_decay_constant_derate_1.0_batchsize_64_hidden_(256, 128)_lambda_0.01
│  ├─epoches_100_lr_0.1_decay_exponential_derate_0.95_batchsize_32_hidden_(256, 128)_lambda_0.01
│  ├─epoches_10_lr_0.01_decay_constant_derate_1.0_batchsize_32_hidden_(256, 128)_lambda_0.005
│  ├─epoches_10_lr_0.1_decay_exponential_derate_0.9_batchsize_32_hidden_(256, 128)_lambda_0.01
│  ├─epoches_30_lr_0.1_decay_constant_derate_1.0_batchsize_32_hidden_(128, 64)_lambda_0.01
│  ├─epoches_30_lr_0.1_decay_exponential_derate_0.95_batchsize_32_hidden_(256, 128)_lambda_0.01
│  ├─epoches_50_lr_0.1_decay_constant_derate_1.0_batchsize_64_hidden_(256, 128)_lambda_0.01
│  └─epoches_50_lr_0.1_decay_exponential_derate_0.95_batchsize_32_hidden_(256, 128)_lambda_0.01
├─fashion-mnist
│  ├─.git
│  │  ├─branches
│  │  ├─hooks
│  │  ├─info
│  │  ├─logs
│  │  │  └─refs
│  │  │      ├─heads
│  │  │      └─remotes
│  │  │          └─origin
│  │  ├─objects
│  │  │  ├─info
│  │  │  └─pack
│  │  └─refs
│  │      ├─heads
│  │      ├─remotes
│  │      │  └─origin
│  │      └─tags
│  ├─benchmark
│  ├─data
│  │  ├─fashion
│  │  └─mnist
│  ├─doc
│  │  └─img
│  ├─static
│  │  ├─css
│  │  ├─img
│  │  └─js
│  ├─utils
│  │  └─__pycache__
│  └─visualization
├─hyperparameter_search_result
├─model_visualization
│  ├─epoches_100_lr_0.1_decay_constant_derate_1.0_batchsize_64_hidden_(256, 128)_lambda_0.01
│  ├─epoches_100_lr_0.1_decay_exponential_derate_0.95_batchsize_32_hidden_(256, 128)_lambda_0.01
│  ├─epoches_10_lr_0.01_decay_constant_derate_1.0_batchsize_32_hidden_(256, 128)_lambda_0.005
│  ├─epoches_10_lr_0.1_decay_exponential_derate_0.9_batchsize_32_hidden_(256, 128)_lambda_0.01
│  ├─epoches_30_lr_0.1_decay_constant_derate_1.0_batchsize_32_hidden_(128, 64)_lambda_0.01
│  ├─epoches_30_lr_0.1_decay_exponential_derate_0.95_batchsize_32_hidden_(256, 128)_lambda_0.01
│  ├─epoches_50_lr_0.1_decay_constant_derate_1.0_batchsize_64_hidden_(256, 128)_lambda_0.01
│  └─epoches_50_lr_0.1_decay_exponential_derate_0.95_batchsize_32_hidden_(256, 128)_lambda_0.01
├─result_plot
│  └─best_search_results
└─__pycache__


### Testing
To test the model, download the model weights from Google Drive:
[Model Weights](https://drive.google.com/drive/folders/1uTwq6ZPBVR2kLWwqin_I10n2ULNaI40I?usp=sharing).

After downloading the weights to your local folder, you can test the model's performance on the test set with the following command: ```python main.py --test-only --model-path model_parameters```


### Training
Set parameters in `config.py` following the comments provided within the file. To train the model, run: ```python main.py```
This command will train the model and automatically save the parameters.

### Hyperparameter Searching
Use the following command for hyperparameter searching, allowing you to set the decay style and number of training epochs: ```python hyperparameter.py --decay_style DS --epochs E```
The default search parameters include learning rate, batch size, hidden layer sizes, regularization strength, and learning rate decay coefficient. For additional adjustments, modifications can be made directly to `hyperparameter.py`.




