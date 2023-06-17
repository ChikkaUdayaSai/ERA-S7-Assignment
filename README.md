# ERA S7 Assignment

This repository contains the solution to the assignment given in The School of AI's ERA Program Session 7.

The problem statement is identification of handwritten digits of MNSIT dataset.

The aim of the assignment is to build a model that can identify handwritten digits from the MNIST dataset with an accuracy of 99.4% or more with below constraints:

    - 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
    - Less than or equal to 15 Epochs
    - Less than 8000 Parameters
    - Do this using your modular code. Every model that you make must be there in the model.py file as Model_1, Model_2, etc.
    - Do this in exactly 3 steps
    - Each File must have a "target, result, analysis" TEXT block (either at the start or the end)
    - You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 
    - Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted. 
    - Explain your 3 steps using these targets, results, and analysis with links to your GitHub files (Colab files moved to GitHub). 
    - Keep Receptive field calculations handy for each of your models. 

## Pre-requisites

The code is written in Python 3.10.11. It is recommended to use a virtual environment to run the code to avoid dependency issues. Try to use Google Colab or Kaggle to run the code as they provide free access to GPUs. If you are running the code on your local machine, make sure you install the virtual environment before running the code.

### Installing the Virtual Environment

It is advised to install Anaconda to manage the virtual environment. Anaconda can be downloaded from [here](https://www.anaconda.com/products/individual). Once installed, the virtual environment can be created using the following command:

```bash
conda create -n era python=3.10.11
```

### Activating the Virtual Environment

The virtual environment needs to be activated before running the code. This can be done using the following command:

```bash
conda activate era
```

## Installation

1. Clone the repository using the following command:

    ```bash
    git clone https://github.com/ChikkaUdayaSai/ERA-S7-Assignment
    ```

2. Navigate to the repository directory:

    ```bash
    cd ERA-S7-Assignment
    ```

3. Install the dependencies using the following commnad:

    ```bash
    pip install -r requirements.txt
    ```

Note: If you are using Google Colab or Kaggle, you can skip the above step as the dependencies are already installed in the environment. But it is advised to check the versions of the dependencies before running the code.

The code uses PyTorch and Torchvision for fetching the MNIST dataset and training the model. An additional dependency, Matplotlib, is used for plotting the training and validation losses. Finally, the Torchsummary package is used to visualize the model architecture.

We are now ready to run the code with the following versions of the dependencies:

- **PyTorch: 2.0.1**
- **Torchvision: 0.15.2**
- **Matplotlib: 3.7.1**
- **Torchsummary: 1.5.1**


## Solution

I have used 6 iterations to solve the problem and 4 different model architectures. The details of each iteration are given below:

### Iteration 1

1. Target:
    
    - Prepare the basic skeleton of the model
    - Make sure the code is working and the model is able to train
    - Try to achieve as much accuracy as possible within 15 epochs


The code for the above architecture can be found in model.py file as Model_1 and the code is as follows:

```python
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x
```

The model architecture is as follows with Receptive field calculations:
    
    1. Convolution Block 1:
        - Input: 28x28x1
        - Output: 28x28x32
        - Receptive Field: 3x3

    2. Convolution Block 2:
        - Input: 28x28x32
        - Output: 28x28x32
        - Receptive Field: 5x5
    
    3. Transition Block 1:
        - Input: 28x28x32
        - Output: 14x14x32
        - Receptive Field: 6x6

    4. Convolution Block 3:
        - Input: 14x14x32
        - Output: 14x14x64
        - Receptive Field: 10x10

    5. Convolution Block 4:
        - Input: 14x14x64
        - Output: 14x14x64
        - Receptive Field: 14x14

    6. Transition Block 2:
        - Input: 14x14x64
        - Output: 7x7x64
        - Receptive Field: 16x16

    7. Convolution Block 5:
        - Input: 7x7x64
        - Output: 7x7x128
        - Receptive Field: 24x24
    
    8. Output Block:
        - Input: 7x7x128
        - Output: 1x1x10
        - Receptive Field: 32x32


2. Results:
        
    - Parameters: 174K
    - Training Accuracy: 99.7%
    - Test Accuracy: 98.34%

3. Analysis:

    - The model is able to achieve almost 99% accuracy within 15 epochs
    - The model is overfitting as the training accuracy is much higher than the test accuracy
    - The model is not able to generalize well as the test accuracy is not increasing after 10 epochs

### Iteration 2

1. Target:
    
    - Reduce the number of parameters
    - Reduce the overfitting
    - Increase the test accuracy

The code for the above architecture can be found in model.py file as Model_2 and the code is as follows:

```python
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 9, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(9, 10, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x

```

    The model architecture is as follows with Receptive field calculations:
    
    1. Convolution Block 1:
        - Input: 28x28x1
        - Output: 28x28x9
        - Receptive Field: 3x3

    2. Convolution Block 2:
        - Input: 28x28x9
        - Output: 28x28x10
        - Receptive Field: 5x5
    
    3. Transition Block 1:
        - Input: 28x28x10
        - Output: 14x14x10
        - Receptive Field: 6x6

    4. Convolution Block 3:
        - Input: 14x14x10
        - Output: 14x14x16
        - Receptive Field: 10x10

    5. Convolution Block 4:
        - Input: 14x14x16
        - Output: 14x14x16
        - Receptive Field: 14x14

    6. Transition Block 2:
        - Input: 14x14x16
        - Output: 7x7x16
        - Receptive Field: 16x16

    7. Convolution Block 5:
        - Input: 7x7x16
        - Output: 7x7x16
        - Receptive Field: 24x24
    
    8. Output Block:
        - Input: 7x7x16
        - Output: 1x1x10
        - Receptive Field: 32x32


2. Results:
        
    - Parameters: 7K
    - Training Accuracy: 99.1%
    - Test Accuracy: 98.9%

3. Analysis:
    
    - The model is able to achieve the target of reducing the number of parameters
    - The model is able to achieve the target of reducing the overfitting
    - However, the model is not able to achieve the target of increasing the test accuracy

### Iteration 3

1. Target:
    
    - Increase the test accuracy
    - Reduce the overfitting, if any

The code for the above architecture can be found in model.py file as Model_3 and the code is as follows:

```python
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 9, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 10, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.transition2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x
```

The model architecture is as follows with Receptive field calculations:
    
    1. Convolution Block 1:
        - Input: 28x28x1
        - Output: 28x28x9
        - Receptive Field: 3x3

    2. Convolution Block 2:
        - Input: 28x28x9
        - Output: 28x28x10
        - Receptive Field: 5x5
    
    3. Transition Block 1:
        - Input: 28x28x10
        - Output: 14x14x10
        - Receptive Field: 6x6

    4. Convolution Block 3:
        - Input: 14x14x10
        - Output: 14x14x16
        - Receptive Field: 10x10

    5. Convolution Block 4:
        - Input: 14x14x16
        - Output: 14x14x16
        - Receptive Field: 14x14

    6. Transition Block 2:
        - Input: 14x14x16
        - Output: 7x7x16
        - Receptive Field: 16x16

    7. Convolution Block 5:
        - Input: 7x7x16
        - Output: 7x7x16
        - Receptive Field: 24x24
    
    8. Output Block:
        - Input: 7x7x16
        - Output: 1x1x10
        - Receptive Field: 32x32

2. Results:
            
    - Parameters: 8K
    - Training Accuracy: 99.34%
    - Test Accuracy: 99.24%

3. Analysis:
        
    - The model is able to achieve the target of increasing the test accuracy by maintaining almost same number of parameters
    - However, the model test accuracy is still fluctuating and not stable and will need to be improved further

### Iteration 4

1. Target:
    
    - Make the test accuracy stable and consistent by making all the layers learn
    - Adding dropout to reduce overfitting

The code for the above architecture can be found in model.py file as Model_4 and the code is as follows:

```python
class Model4(nn.Module):
    def __init__(self):
        DROP = 0.01
        super(Model4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 9, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 10, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.transition1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.transition2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x
```

The model architecture is as follows with Receptive field calculations:
    
    1. Convolution Block 1:
        - Input: 28x28x1
        - Output: 28x28x9
        - Receptive Field: 3x3

    2. Convolution Block 2:
        - Input: 28x28x9
        - Output: 28x28x10
        - Receptive Field: 5x5
    
    3. Transition Block 1:
        - Input: 28x28x10
        - Output: 14x14x10
        - Receptive Field: 6x6

    4. Convolution Block 3:
        - Input: 14x14x10
        - Output: 14x14x16
        - Receptive Field: 10x10

    5. Convolution Block 4:
        - Input: 14x14x16
        - Output: 14x14x16
        - Receptive Field: 14x14

    6. Transition Block 2:
        - Input: 14x14x16
        - Output: 7x7x16
        - Receptive Field: 16x16

    7. Convolution Block 5:
        - Input: 7x7x16
        - Output: 7x7x16
        - Receptive Field: 24x24
    
    8. Output Block:
        - Input: 7x7x16
        - Output: 1x1x10
        - Receptive Field: 32x32

2. Results:
                
    - Parameters: 8K
    - Training Accuracy: 99.24%
    - Test Accuracy: 99.26%

3. Analysis:

    - The overall model accuracy is stable and consistent
    - The model is not overfitting and the gap between training and test accuracy is very less
    - We can consider this as the best model so far and can be used for further analysis


### Iteration 5

1. Target:
    
    - See how image augmentation can help in improving the model accuracy

We are using the same model as Model_4 and adding image augmentation to the training dataset.

2. Results:
                
    - Parameters: 8K
    - Training Accuracy: 98.69%
    - Test Accuracy: 99.53%

3. Analysis:

    - The overall model accuracy is stable and consistent
    - The training accuracy has reduced but the test accuracy has increased
    - Need to experiment with LR and other hyperparameters to see if we can improve the training accuracy

### Iteration 6

1. Target:
    
    - See how LR Scheduler can help in improving the model accuracy
    - The scheduler used here is ReduceLROnPlateau

We are using the same model as Model_4 and adding ReduceLROnPlateau scheduler during the training.

2. Results:
                
    - Parameters: 8K
    - Train Accuracy = 98.85
    - Test Accuracy = 99.44

3. Analysis:

    - The overall model accuracy is stable and consistent
    - The training accuracy has reduced but the test accuracy has increased
    - We have achieved the target of 99.4% test accuracy in last 2 iterations.




