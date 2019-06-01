# Classification-With-HOG-K-mean-Logistic

## Intuition
  - Build a Classifier to classify metropolitian & countryside
  - Using:
      - K-Means clustering, histogram (HOG) for prerocess data and extract features
      - LogisticsRegression for classify metropolitian and countryside
  - Data: [metropolitian](https://drive.google.com/open?id=1xjVkboqPbmeEnXPyN7D2tpNRxAOhdhEa) & [countryside](https://drive.google.com/open?id=1kJjjszN0nv5y2xfn3TWqVMJSu6aAa5K5)

## Architecture
  - Preprocess Images : Using opencv for imread data with function [load_image](https://github.com/minhhaui/Classification-With-HOG-K-mean-Logistic/blob/master/preprocess_image.py#L8) and resize output image (350,350)
  
  - K-Means Clustering : We will extract dominant colors with k = 5 centroid with function [setUPKmean](https://github.com/minhhaui/Classification-With-HOG-K-mean-Logistic/blob/master/preprocess_image.py#L35)
  
  - HOG (Histogram Oriented Gradients) : Construct the features vector for each image using 5 clustered RGB colors, sorted by color frequency.
  
  - LogisticRegression : We use [Logistic Regression](https://github.com/minhhaui/Classification-With-HOG-K-mean-Logistic/blob/master/setup_model.py#L6) model to classify the feature vectors built from the HOG.
    Implement the Gradient Descent method to optimize the model.
       ```
        batch_size = 32
        epochs = 10
        threshold = 0.5
        learning_rate = 0.001
        
       ````
  - Accuracy: 73.8 % with 1206 samples ( validation set = 20% , training set = 80%)
  - Output K-means Clustering with centroid = 5:
  
     <img src = "https://github.com/minhhaui/Classification-With-HOG-K-mean-Logistic/blob/master/resources_data/output_image_test/kmean.png">
   - Vector Histogram:
   
     <img src = "https://github.com/minhhaui/Build-Classifier-LogisticRegression/blob/master/resources_data/output_image_test/Screenshot%20from%202019-06-01%2022-47-59.png">
   - Ouput HOG:  
   
     <img src = "https://github.com/minhhaui/Classification-With-HOG-K-mean-Logistic/blob/master/resources_data/output_image_test/hog.png">
     
  
# Tree Project
```
  Classification/
                model/
                      model.sav # model results after training
                resources_data/
                        countryside         # contains data of countryside ( have some image for intuition)
                        metropolitian       # contains data of metropilitian ( have some image for intuition)
                        image_predict       # contains image for test predict
                        output_image_test   # some results of k-mean and hog
                predict.py                  # predict some image from resources_data/image_predict with args pathto model
                preprocess_image.py         # contains imread data, fit k-mean and hog
                setup_model.py              # Build LogisicRegression from scatch
                train.py                    # start training with args path data train
                      
```

# USAGE:
## Requirements
   ```
      skimage
      opencv-python
      numpy
      matplotlib
      sklearn
   ```
## Training
  `python train.py --inputcountry path_to_countrydata/ --inputmetro path_to_metropolitiandata/`
## prediction
  `python predict.py --model path_to_model/ --image path_to_image/`
