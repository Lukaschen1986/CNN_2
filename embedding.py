from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Flatten, Dense
from keras import initializers
from keras.regularizers import l2
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
from keras.models import load_model, model_from_json, model_from_yaml
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation

input_dim, input_length = x_train.shape
latent_dim = 128

x_input = Input(shape=(input_length,), dtype=np.int32, name="x_input")

embedding = Embedding(input_dim=input_dim,
                      output_dim=latent_dim,
                      embeddings_initializer=initializers.TruncatedNormal(0.0, 0.01),
                      embeddings_regularizer = l2(0.0),
                      input_length=input_length,
                      name="embedding")
mlp_vector  = Flatten()(embedding(x_input))

mlp_vector = Dense(units=256, 
                   activation=None, # softmax, sigmoid, relu
                   use_bias=False,
#                   bias_initializer=initializers.zeros(),
                   kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                   name="layer_1")(mlp_vector)
mlp_vector = BatchNormalization(axis=1, center=True, scale=True, 
                                beta_initializer=initializers.zeros(), 
                                gamma_initializer=initializers.ones(), 
                                epsilon=10**-8, 
                                momentum=0.9)(mlp_vector)
mlp_vector = Activation("relu")(mlp_vector)

mlp_vector = Dense(units=512, 
                   activation=None, # softmax, sigmoid, relu
                   use_bias=False,
#                   bias_initializer=initializers.zeros(),
                   kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                   name="layer_2")(mlp_vector)
mlp_vector = BatchNormalization(axis=1, center=True, scale=True, 
                                beta_initializer=initializers.zeros(), 
                                gamma_initializer=initializers.ones(), 
                                epsilon=10**-8, 
                                momentum=0.9)(mlp_vector)
mlp_vector = Activation("relu")(mlp_vector)

prediction = Dense(units=1, 
                   activation=None, # softmax, sigmoid
                   use_bias=False,
                   bias_initializer=initializers.zeros(),
                   kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                   name="prediction")(mlp_vector)
prediction = BatchNormalization(axis=1, center=True, scale=True, 
                                beta_initializer=initializers.zeros(), 
                                gamma_initializer=initializers.ones(), 
                                epsilon=10**-8, 
                                momentum=0.9)(prediction)
prediction = Activation("sigmoid")(prediction)

model = Model(inputs=x_input, outputs=prediction)
model.summary()

plot_model(model, to_file="D:/my_project/Python_Project/iTravel/virtual_quota_room/txt/model/model.png", show_shapes=True, show_layer_names=True)

bs = 128; epc = 3; lr = 0.1; dcy = 0.01
model.compile(loss="binary_crossentropy", 
              optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=dcy), # , decay=dcy
              metrics=["binary_accuracy"])
early_stopping = EarlyStopping(monitor="loss", patience=4, mode="min", verbose=1)
model_fit = model.fit(x_train, y_train, batch_size=bs, epochs=epc, verbose=1, shuffle=True, callbacks=[early_stopping])

model.save("D:/my_project/Python_Project/iTravel/virtual_quota_room/txt/model/model.h5", overwrite=True, include_optimizer=True)
model = load_model("D:/my_project/Python_Project/iTravel/virtual_quota_room/txt/model/model.h5", compile=True)

layer_2 = model.get_layer("layer_2").get_weights() # get_weights
model.get_layer("layer_2").set_weights(layer_2) # set_weights

model_layer_2 = Model(inputs=model.input, 
                      outputs=model.get_layer("layer_2").output)
output_layer_2 = model_layer_2.predict(x_train)
output_layer_2_test = model_layer_2.predict(x_test)
