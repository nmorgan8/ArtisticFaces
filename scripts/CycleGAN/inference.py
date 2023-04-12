
import tensorflow as tf
import numpy as np
import keras.utils as image
tf.compat.v1.disable_eager_execution() #[C]:Placing this code block here to because of keras upgrade
from keras.applications.resnet import preprocess_input, decode_predictions, ResNet50
from VGG_model import *
from preprocessing import *
from keras.models import Model
from keras import backend as K
from dataetime import datetime
import matplotlib.pyplot as plt  
import scipy  

#loading,processing and defining multi_output_model and style loss computation of style image
style_img_path = '/content/Abstract Style Art.jpeg'
person_img_path = '/Users/claudiagusti/Desktop/Spring2022/CMSI 6998/ArtisticFaces/scripts/VGG/sample_img/LinkedinPhot.jpeg'
img = image.load_img(path)
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)    
x = preprocess_input(x)

#shape
batch_shape = x.shape
shape = x.shape[1:]
vgg = vgg_avg_pooling(shape)

#define multi-output model
symb_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')]
multi_output_model = Model(vgg.input, symb_conv_outputs)
symb_layer_out = [K.variable(y) for y in multi_output_model.predict(x)]

#Conv layer weight matrix
weights = [0.2,0.4,0.3,0.5,0.2]    
loss=0
#Total style loss
for symb,actual,w in zip(symb_conv_outputs,symb_layer_out,weights):
    loss += w * style_loss(symb[0],actual[0])

#gradients which are needed by the optimizer    
grad = K.gradients(loss,multi_output_model.input)
#.function should be lower case 
get_loss_grad = K.function(inputs=[multi_output_model.input], outputs=[loss] + grad)

#Scipy's minimizer function(fmin_l_bfgs_b) allows us to pass back function value f(x) and 
#its gradient f'(x), which we calculated in earlier step. 
#However, we need to unroll the input to minimizer function in1-D array format and both loss and gradient must be np.float64.
def get_loss_grad_wrapper(x_vec):
    """Scipy's minimizer function(fmin_l_bfgs_b) allows us to pass back function value f(x) and 
       its gradient f'(x), which we calculated in earlier step. 
       However, we need to unroll the input to minimizer function in1-D array format and both loss and gradient must be np.float64."""
    l,g = get_loss_grad([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64)

#Function to minimize loss
def min_loss(fn,epochs,batch_shape):
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = scipy.optimize.fmin_l_bfgs_b(func=fn,x0=x,maxfun=20)
    # bounds=[[-127, 127]]*len(x.flatten())
    #x = np.clip(x, -127, 127)
    # print("min:", x.min(), "max:", x.max())
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)
    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()

    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0]