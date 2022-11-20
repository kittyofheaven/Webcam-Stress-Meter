# Webcam Stress Meter
Submitted to MetroHacks2022 **under 18**

## Inspiration
While surfing around google and youtube I found this interesting article about [Photoplethysmogram](https://en.wikipedia.org/wiki/Photoplethysmogram). It tells us that traditional oxy/heart rate monitor use cameras to measure changes in light absorption (**corresponding to an absorption peak by (oxy-) hemoglobin**) that can be converted to a heart rate. So I think why not use a webcam? While it looks very impossible but it's possible by setting the webcam for high iso sensitivity and capturing a color fluctuation on our skin.  

And secondly, there is an article where it states that our heart rate is related to the stress we experience [Effects of stress on heart rate complexity—A comparison between short-term and chronic stress](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2653595/). So after a bit of brainstorming, I decided to build Webcam Stress Meter.  


## What it does
It calculates your heart rate only using your webcam and combines it using the face emotions model resulting in a **calculation for the stress meter**. You can use it while you working/studying so that when your stress meter is reaching a high level, the app will alert you and recommend you to take a rest.

## How we built it
To build the webcam heart rate monitor I use cv2 to capture, and distribute webcam data, some filters, and to track our faces. Then a numpy gives some calculation to the tensor so that the app can measure the light change in our skin and extract it into a heart rate.

For the face emotions model, I use this [kaggle data](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) and train it using google colab with TensorFlow (CNN). then import it to the script using TensorFlow again and with help of cv2, the app can predict a real-time webcam stream.

Then I combine this two to calculate the stress level of the users using a basic math & numpy.
## Challenges we ran into
it's challenging to integrate the heart rate app and face emotions app, especially when I have to make a function to determine the stress level, but it is all done now, and it's working :))

## Accomplishments that we're proud of
complete this app only for 24 hours

## What's next for Webcam Stress Meter
Create an exe installation package for this app, or maybe publish this app as a mobile app in App Store / Play Store 

