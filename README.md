# Caltech UCSD Birds 200 Project
![CUB](https://github.com/user-attachments/assets/1b8e8490-86c0-48b0-b9fb-e67f61198c4f)
## Objective
* The objective of this assignment is to develop a deep learning model for classifying bird species. Your task involves devising your own strategies to train the model, which includes making decisions on data preprocessing, model architecture, and learning objectives. You are encouraged to explore various state of the art techniques in semi-supervised learning, self-supervised learning, and transfer learning to effectively train your model.
## Dataset
#### Caltech UCSD Birds 200 Dataset (CUB-200)
* Specifically designed for bird species classification tasks.
* Contains 11,788 images across 200 different bird species.
## model Architecture
* **Self-attention**: Learns global image relationships (overall structure, patterns) to determine “where to look”
* **SEBblock**: Adjusts channel-wise importance to determine “what to look at”
* **Bottleneck block**: Provides stability and efficiency for ResNet training
* **Full ResNet Structure**: Gradually builds higher-level features to perform final classification
## Final Performance
* Accuracy on CUB validation images: 23.27%
