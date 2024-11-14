# Chest X-ray Classification for COVID-19 Detection

## Overview
This project focuses on classifying chest X-ray images to detect COVID-19, bacterial pneumonia, and normal cases using deep learning techniques. We explored custom CNN architectures and transfer learning with ResNet50, addressing challenges like data imbalance and dataset variability to improve classification accuracy.

## Project Goals
1. **Data Exploration and Preparation**: Analyze the data distribution and address dataset inconsistencies.
2. **Model Training**: Develop a CNN model from scratch and experiment with ResNet50 for transfer learning.
3. **Address Data Imbalance**: Test methods such as oversampling, undersampling, and weighted loss functions to handle the imbalanced classes.
4. **Evaluate Performance**: Assess model performance using metrics such as accuracy, precision, and per-class accuracy, with special focus on COVID-19 classification accuracy.

## Dataset
The dataset includes chest X-ray images across three classes:
- **Normal**
- **Bacterial Pneumonia**
- **COVID-19 Pneumonia**

## Methodology

### 1. Data Preprocessing
- **Image Resizing**: All images were resized to 256x256 and converted to grayscale.
- **Contrast Normalization**: Applied histogram equalization and normalization for consistent image contrast.
- **Data Augmentation**: Techniques like random rotation, flipping, and shifting were used to improve robustness.

### 2. Handling Class Imbalance
We tackled the class imbalance problem through:
- **Oversampling**: Increasing instances of the minority class.
- **Undersampling**: Reducing instances of the majority class.
- **Class Weights**: Adjusting the loss function to give higher weight to the minority class.

### 3. Test Time Augmentation (TTA)
To improve test performance, we applied multiple augmentations at inference, including rotations and flips, and averaged predictions across augmented versions to increase robustness.

## Results
- **Best Performance**: The over-sampled dataset yielded the best results, especially in improving the COVID-19 classification accuracy.
- **Transfer Learning**: Fine-tuned ResNet50 performed comparably with our custom CNN model, offering a stable solution with less overfitting.
- **Test Time Augmentation**: Enhanced model robustness, resulting in more balanced performance across classes.

## Conclusion
This project demonstrates the application of deep learning for medical image classification with effective techniques to handle data variability and class imbalance. Future improvements could include more sophisticated model architectures and advanced fine-tuning methods.

