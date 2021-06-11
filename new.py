import pickle
import numpy as np
import keras

model_path = 'static/pickle/holly_MobileNet_3(50_class).h5'
label_path = 'static/pickle/holly_50_classes_lableencoder.pickle'

new_pickle = 'static/pickle/val1622412457_50_class.pickle'

model = keras.models.load_model(model_path)

# r+ is used for reading b is used for binary
labelEncoder = pickle.load(open(label_path,'rb'))

pick=pickle.load(open(new_pickle,'rb'))
print(pick)

count = 0
c = 0
print("Printing the first 100 test cases......")

for img,actual in pick:
    c+=1
    img1 = img
    
    img = img.astype(np.float32) / 255.0
    np_img = img[np.newaxis,:, :,:]
    preds = model.predict(np_img)
    out = np.argmax(preds)
    name = labelEncoder.get(out)
    if actual == out:
        count += 1
print(count/c)
    
'''c+= 1
    if c <= 100:
        #cv2_imshow(img1)
        print('Actual = ', labelEncoder.get(actual)[5:], ' **** Predicted = ', name[5:])
        print('-------------------------------------------------------------------\n\n\n')'''


