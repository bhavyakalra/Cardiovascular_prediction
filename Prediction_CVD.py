import numpy as np

import pickle





loaded_model = pickle.load(open('Model_rf.pickle', 'rb'))

input_data = (3,2,0,0,0,0,0,1,1,150.0,32.66,14.54,0,30,16,12,4,1.0,0.0,0.0,1.0)

input_data_np = np.asarray(input_data)

input_data_reshaped = input_data_np.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)

print(prediction)

if (prediction[0] == 0):
    print('Person is not prone to Heart Disease')
else:
    print('Person is prone to Heart disease')