# HCNNs_and_Chaos
In this repo, you will find  all the codes that were used in various experiments for my PhD. This includes HCNN and RNN modules for the modeling of chaotic systems. 
## Code Implementation Overview

The development and implementation of the Historical Consistent Neural Networks (HCNN) and their various long-memory improvement methods were carried out entirely in the *PyTorch* library, leveraging its flexibility and efficiency for deep learning tasks. This section provides an overview of the key aspects of the codebase, detailing the structure, functionality, and design principles behind the models, data handling, and training processes. The code is available in the GitHub repository at [HCNN_and_Chaos](https://github.com/rock-feller/HCNNs_and_Chaos).

### PyTorch-Based Implementation

All the codes were implemented using *PyTorch*, where we created custom modules for our HCNN model classes. The HCNN models, including their memory improvement methods, were written from scratch. This includes the definition of the forward and backward methods, which dictate how the data flows through the network and how gradients are computed during backpropagation. Custom training functions were also implemented to handle the specific requirements of HCNN models, ensuring efficient and effective learning. An example of the Vanilla HCNN module is provided below.

### Modeling Frameworks

Two distinct modeling frameworks were developed to address different types of systems:

**HCNNs_Chaos**: This framework contains all the modules required for modeling the deterministic systems, including the Lorenz, Rossler, and Rabinovich-Fabrikant systems. It includes 4 folders:

- **data**: All the different custom logics for data generation can be found here.
- **data_prep**: All the modules for data preprocessing can be found here, including normalization and transformation.
- **hcnn_modules**: The different HCNN modules can be found here.
- **modeling_strategy**: This folder contains all the different custom training functions.

A similar folder for RNN-based models (named **RNNs_Chaos**) can also be found at the same location.

**HCNNs_Climate**: This framework contains all the modules required for modeling partially observable systems. It includes 3 folders:

- **data_prep**: All the modules for data preprocessing can be found here, including sliding windows, normalization, and transformation.
- **hcnn_modules**: The different HCNN modules can be found here.
- **modeling_strategy**: This folder contains all the different custom training functions.

A similar folder for RNN-based models (named **RNNs_Climate**) can also be found at the same location.

### Training and Results Management

To streamline the training process and facilitate the analysis of results, a structured approach was implemented for managing the outputs of the training sessions:

**Output Folders:** At the end of each training session, a results folder is automatically created. This folder is organized into three sub-folders, each serving a specific purpose:

- **output_dicts:** This sub-folder contains JavaScript Object Notations (JSON) files that are named according to the model names. Each JSON file tracks important metrics such as training loss, best test loss, and the time taken for each epoch. This detailed tracking allows for a comprehensive analysis of the models' performance and the computational resources required.
- **csv_files:** This sub-folder stores Comma Separated Values (CSV) files that record the forecasted values in a tabular format. These CSV files are essential for preserving the predicted values and are particularly useful for subsequent plotting and visualization of the results.
- **trained_models:** The serialized versions of the trained models (*state_dicts*) are saved in this sub-folder as `.pt` files. This allows for easy reusability of the models, enabling additional training sessions or deployment without the need to retrain from scratch.

### Data Handling and Preprocessing

The data used for training and evaluation were loaded and manipulated as tensors, enabling scalable and efficient computations. Given the complexity of the tasks and the volume of data, several preprocessing steps were designed and implemented.

The preprocessing pipeline includes methods for normalization, scaling, cleaning, and plotting. These operations were crucial for transforming raw data into a format suitable for input into the HCNN models. The logic for these preprocessing steps was carefully designed to ensure that the data fed into the network maintained its integrity and relevance for the modeling tasks.

### Optimization for GPU and Apple Silicon

To maximize computational efficiency, the codebase is optimized to run on both Compute Unified Device Architecture (CUDA-enabled) GPUs and Apple Silicon Metal Performance Shaders (MPS) devices. This optimization ensures that the training and evaluation processes can fully utilize the available hardware, significantly reducing the time required for model training, especially when dealing with large datasets and complex models. By supporting CUDA and MPS, the code provides flexibility in deployment across various hardware platforms, making it accessible for a wide range of users with different computational resources.

