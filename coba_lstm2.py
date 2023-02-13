import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from matplotlib import pyplot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

df = pd.read_csv('D:/TESIS HANA/data/alpari/EURUSD_5M_2008-2016processed_lstm_barupj_Daily_bener.csv')
df.dropna(inplace=True)
df = df.iloc[:49,:]
print(df)

#df_train = df.iloc[:51760,:]
#df_test = df.iloc[51760:,:]

def data(df,n_steps):
    #arr_high = np.array(df['high'].to_list())
    #arr_low = np.array(df['low'].to_list())
    #arr_close = np.array(df['close'].to_list())
    #arr_open = np.array(df['open'].to_list())
    arr_color = np.array(df['color'].to_list())
    arr_action = np.array(df['action'].to_list())
    #arr_action = tf.keras.utils.to_categorical(arr_action, num_classes=3)
    arr_profit = np.array(df['profit'].to_list())
    arr_loss = np.array(df['loss'].to_list())
    arr_label = np.array(df['label'].to_list())
    #arr_color1h = np.array(df['color1h'].to_list())
    arr = df[['color','action','PL']].values
    #arr_subtract_co = np.subtract(arr_close,arr_open)
    #arr_subtract_hl = np.subtract(arr_high,arr_low)
    arr_sum = np.add(arr_profit,arr_loss)
    #arr_multiply = np.multiply(arr_subtract_co,arr_subtract_hl)
    arr3d = []
    arrLabel = []
    for i in range(n_steps,len(arr_label)-n_steps):#94):
        #temp = []
        #for j in range(i-48,i):#(i-47,i+1): 
        #    temp.append([arr_color[j],arr_action[j],arr_sum[j]])
        arr3d.append(arr[i-n_steps:i])
        arrLabel.append(arr_action[i])
    arr3d = np.array(arr3d)
    arrLabel = np.array(arrLabel)#arr_action[n_steps:])
    return arr3d,arrLabel

n_steps = 48
arr3d,arrLabel = data(df,n_steps)
#arr3d = np.array(arr3d)
#arrLabel = np.array(arrLabel)
#train_X,test_X, train_y, test_y = train_test_split(arr3d, arrLabel, test_size=0.5,shuffle=False)
test_X,test_y = arr3d, arrLabel

#train_X, train_y = data(df_train)
#test_X, test_y = data(df_test)
#train_y = tf.keras.utils.to_categorical(train_y, num_classes=3)
test_y = tf.keras.utils.to_categorical(test_y, num_classes=3)

print(test_y)
#print(train_y)
n_features = 3
#train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))
#print(train_X)
print(test_X)
'''#to not get random result.
np.random.seed(123)
random.seed(123)
tf.random.set_seed(1234)
#LSTM model
model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
#model.add(tf.keras.layers.LSTM(100, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(16))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#model = tf.keras.models.load_model('./bener/1training1layer_100_16_64unit_color_profitloss_2015_pj.h5')
model.summary()
#checkpoint_path = "./bener/1training1layer_50_16_64unit_tanpaclose_profitlossditambah_2008-2016_baru_dropout/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                 save_weights_only=True,
#                                                 verbose=1)
# fit network
tf.keras.utils.plot_model(
model,
to_file="model.png",
show_shapes=True,
show_dtype=False,
show_layer_names=True,
rankdir="TB",
expand_nested=True,
dpi=96,
layer_range=None,
show_layer_activations=True,
)
#history = model.fit(train_X, train_y, epochs=20, batch_size=16,validation_data=(test_X, test_y), verbose=2, shuffle=False)#,callbacks=[cp_callback])

#model.save('./bener/1training1layer_20_16_64unit_2008-2016_barupj_1steplabel_color3pips_500_steps_bener.h5')
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
'''
# Loads the weights
#model.load_weights(checkpoint_path)
#print('test x',test_X)
#print('test y',test_y)
# Re-evaluate the model
policy = tf.keras.models.load_model("D:/TESIS HANA/code thesis/lstm_exp/bener/1training1layer_20_16_64unit_2008-2016_barupj_1steplabel_color3pips.h5")

#loss, acc = model.evaluate(test_X, test_y)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
predict = policy.predict(test_X)
print(predict)