# hunchback-detection 驼背检测
A lightweight sitting posture classifier using frontal camera images. 
You can optionally send a Windows notification to remind you. 
This is what I used personally to help correct my sitting posture. 
The training data describes my own behavior, 
you may need to collect your own data and train your own model (takes about 2 minutes).

## Dependencies

```
opencv-python
mediapipe
scikit-learn
win10toast (for poping Windows notification)
```

## Data Collection

Run `datacollect.py`, after seeing your images, maintaining an 'excellent' posture while pressing the key A. 
Then switch to an 'okay' posture while pressing the key B.
Then switch to an 'bad' posture while pressing the key C.
Then switch to an 'terrible' posture while pressing the key D.
Data instance count will be shown on the terminal. 400 instances for each category works well for me.
After collecting, press ESC to exit the program and a `posture_data.npy` file will be saved.

## Model Training

Run `trainclassifer.py` to train the model, a simple scikit-learn LogisticRegression classifier.
The model will be saved as `model.sav`

## Demo

To see how well your model is, run `predict_demo.py`. The classified posture will be shown as text on screen.

## Actual Usage

Run `predict_nofity.py` to use the actual posture remainder program. 
The classifier will run in the background and pop up a Windows notification in the lower right corner 
whenever you are in a bad sitting posture.


