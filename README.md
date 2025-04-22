# CAA Project 1 - Garbage Classification

Project 1 of CAA UA 2025

## Instruction 

- First run the preprocess.ipynb file to download the files

## [Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/code?datasetId=81794&sortBy=voteCount)

## [Report](https://typst.app/project/ro0ldpqhMWggBg9W2F50nf)

## State of the art
The issue of waste classification has been widely discussed due to the large amount of waste produced by man and its environmental effects, such as pollution and health-related risks [1]. In addition, landfills, for example, take up physical space and have the capacity to cause atmospheric, hydrosphere, and geosphere pollution, and burning can lead to the emission of contamination into the air [2].
Therefore, it is essential to study waste classification, the automatic distinction between different types of waste, and consequently, the environmental and social impact it has on the planet. In this sense, we initially sought to learn a little about previous studies and the respective architectures used. 
That said, several articles were found that sought to explore the topic and in which architectures such as Convolutional Neural Networks (CNNs), MobileNet, and ResNet, among others, were tested.
The articles “Garbage Classification Using Deep Learning” and “Deep Learning Approach to Recyclable Products Classification: Towards Sustainable Waste Management” are examples of articles in which a system that uses CNNs is used. In this case, for the first article, a garbage classification algorithm using computer vision was proposed that aimed to classify garbage into organic, recyclable and non-recyclable, achieving an accuracy rate of 95% [1]. The second article also suggested an approach for categorising recyclable products. The ScrapNet model was used, and transfer learning techniques were applied with the pre-trained InceptionV3 model. Finally, data augmentation techniques were used to improve performance, and an accuracy of 92.87% was achieved [3].
However, as mentioned, some examples sought to use different approaches, such as the article “DeepWaste: Applying Deep Learning to Waste Classification for a Sustainable Planet”, which proposed DeepWaste (a mobile application for real-time waste classification) using ResNet50. This achieved 88.1% accuracy in tests with authentic images [4]. Also noteworthy are the articles “Fine-Tuning Models Comparisons on Garbage Classification for Recyclability” and “Advancing Recycling Efficiency: A Comparative Analysis of Deep Learning Models in Waste Classification”, in which several models are tested and compressed. For example, the fine-tuning model comparison study compares networks such as AlexNet, VGG16, GoogLeNet and ResNet on a dataset with 2527 images from 6 categories [5]. Furthermore, it uses transfer learning and evaluates the performance using both Softmaz and SVM, achieving an accuracy of 97.86% with ResNet + SVM [5]. The Advancing Recycling Efficiency study compares simple CNNs, AlexNet, ResNet, ResNet50 combined with SVM, and even Transformers to evaluate the efficiency in waste classification. The study shows that more advanced models (such as ResNet50+SVM and Transformers) offer clear improvements in accuracy, reaching 95% [6].
In short, the literature review shows that using AlexNet, VGG, and even ResNet networks is common, which take advantage of small and more delimited datasets, such as TrashNet. Although the results are encouraging, most works do not perform a very in-depth evaluation, and not all attempt to use transfer learning or even more elaborate data augmentation techniques. In this context, this work aims to compare different approaches to optimize accuracy and promote the development of more intelligent and flexible tools for automatic waste management.



## Data Analysis
### 1.	Data Description
The chosen dataset concerns a data set related to waste classification. It is divided into images of waste classified by different categories, such as cardboard, glass, metal, paper, plastic, and trash.
The dataset in question contains about 2,500 images in total, with each class having approximately 400 to 500 images. Table I shows the total number of images per category and an illustrative image of each class. It should also be noted that the images have different sizes but are generally small and of moderate resolution, which is common in simple image classification datasets.

Table I

### 2. Data preprocessing
After performing data analysis, it is essential to understand the dataset to be used in the training process. Thus, data visualization helps us understand, interpret, and extract insights from data intuitively and quickly, as well as identify outliers and any features that may not be relevant for training. 
That said, Figure 7 constructs a graphical visualization of the distribution of images by class, from which it is possible to observe that the number of images is not uniformly distributed among the different categories of waste (cardboard, glass, metal, paper, plastic, trash). Furthermore, this disproportionality may introduce challenges in model training, such as bias towards classes with more examples.

Figure 7

Early identification of this imbalance is essential, as it may justify using techniques such as oversampling, undersampling or class weighting during model training. 

In addition, samples of images belonging to various classes were also viewed, as shown in Figure 8. This visualization allowed us to conclude the visual diversity within the same class, the presence of noise in the images, such as complex backgrounds or partially visible objects, and some differences in lighting and capture angle. In other words, it was possible to identify some details that could impact the generalization capacity of the models.

Figure 8

### 3.	Data preprocessing
After analyzing the dataset in question, applying a set of data preprocessing steps was essential to guarantee data homogeneity. 
Therefore, to ensure that the input for the models had fixed dimensions, all images were first resized to a standard resolution of 224 x 224 pixels. Next, they were converted and normalized using the ToTensor() function of the torchvision library. In this way, the images were converted into arrays of floats, and the pixel values were automatically normalized to the interval [0,1]. 
Finally, the dataset was divided into two subsets: the training and testing sets. In this case, approximately 80% (2021 examples) of the data was split into training and 20% into testing (506 examples), a common proportion.



## References
[1] I. Joshi, P. Dev and G. Geetha, "Garbage Classification Using Deep Learning," 2023 International Conference on Circuit Power and Computing Technologies (ICCPCT), Kollam, India, 2023, pp. 809-814, doi: 10.1109/ICCPCT58313.2023.10245133.
[2] Eurostar, “Waste statistics”, Eurostat Statistics Explained, Apr. 2024. [Online]. Available: https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Waste_statistics
[3] Ahmed, M. I. B., Alotaibi, R. B., Al-Qahtani, R. A., Al-Qahtani, R. S., Al-Hetela, S. S., Al-Matar, K. A., Al-Saqer, N. K., Rahman, A., Saraireh, L., Youldash, M., & Krishnasamy, G. (2023). Deep Learning Approach to Recyclable Products Classification: Towards Sustainable Waste Management. Sustainability, 15(14), 11138. https://doi.org/10.3390/su151411138
[4] Y. Narayan, “DeepWaste: Applying Deep Learning to Waste Classification for a Sustainable Planet,” arXiv preprint arXiv:2101.05960, Jan. 2021. [Online]. Available: https://arxiv.org/abs/2101.05960
[5] X. Zhang, L. Guo, J. Xie, and S. Wang, “Fine-Tuning Models Comparisons on Garbage Classification for Recyclability,” arXiv preprint arXiv:1908.04393, Aug. 2019. [Online]. Available: https://arxiv.org/abs/1908.04393
[6] A. Ahmed, M. Said, and M. Chowdhury, “Advancing Recycling Efficiency: A Comparative Analysis of Deep Learning Models in Waste Classification,” arXiv preprint arXiv:2411.02779, Nov. 2024. [Online]. Available: https://arxiv.org/abs/241
[7] asdasdasasdas, "Garbage classification," Kaggle, Dataset, 2020. [Online]. Available: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification. [Accessed: Apr. 22, 2025].




## Authors

- Diogo Machado Marto NºMec 108298
- Inês Matos NºMec 124349
