# DeepFM论文复现
# （1）导入数据，preprocessing，得到feature_index,feature_value
import pandas as pd
import config
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib as plt
TfTrain = pd.read_csv("/Users/taijieshengwu/PycharmProjects/tensorflow-DeepFM-master/example/data/train.csv")
TfTest = pd.read_csv("/Users/taijieshengwu/PycharmProjects/tensorflow-DeepFM-master/example/data/test.csv")

cols = [i for i in TfTrain.columns if i not in config.IGNORE_COLS]
# 37columns not 39 because we not do messing_value and inner product

def feature_dictionary(TfTest,TfTrain,cols):
    ct = 0
    fd = {}
    df = pd.concat([TfTest,TfTrain])
    X_train = TfTrain[cols].values
    y_train = TfTrain["target"].values
    for col in df.columns:
        if col in config.IGNORE_COLS:
            continue
        elif col in config.NUMERIC_COLS:
            fd[col] = ct
            ct += 1
        else:
            categories = df[col].unique()
            fd[col] = dict(zip(categories,range(ct,ct+len(categories))))
            ct += len(categories) 
    feature_size = ct  #257
    field_size = len(cols) #37
    return fd,feature_size,field_size,X_train,y_train
fd,feature_size,field_size,X_train,y_train = feature_dictionary(TfTest,TfTrain,cols)

def parse(df=None,fd=None,has_label=False):
    if(has_label):
        y_train = df["target"].values.tolist()
    else:
        ids_test = df["id"].values.tolist()
    feature_value = df.copy()
    feature_index = feature_value.copy()
    # feature_index
    for col in df.columns:
        if col in config.IGNORE_COLS:
            feature_index.drop(col,axis = 1,inplace = True)
            feature_value.drop(col,axis=1,inplace=True)
            continue
        elif col in config.NUMERIC_COLS:
            feature_index[col] = fd[col]
        else:
            feature_index[col] = feature_index[col].map(fd[col])
            feature_value[col] = 1
    v_ = feature_value.values.tolist()
    i_ = feature_index.values.tolist()

    feature_value = tf.cast(feature_value,tf.float32)
    feature_index = tf.cast(feature_index,tf.int32)
    if(has_label):
        
        return feature_index,feature_value,y_train,v_,i_
    else:
        return feature_index,feature_value,ids_test,v_,i_
Xi,Xv,y,Xi_train,Xv_train = parse(TfTrain,fd)
Xi_test,Xv_test,ids_test,Xi_test_list,Xv_test_list = parse(TfTest,fd)


def init_weight(deep_layers,feature_size,embedding_size):
    weights = dict()
    # embedding
    weights["feature_embedding"] = tf.Variable(
    tf.random_normal([feature_size,embedding_size],0.0,0.01),
                                           name="feature_embedding") # None* feauture_size*K  #(257, 8)
    print("feature_embedding",weights["feature_embedding"].shape)

    # FM component
    weights["feature_bias"] = tf.Variable(
    tf.random_uniform([feature_size,1],0.0,0.01),
    name="feature_bias")
   
    # deep component
    num_layers = len(deep_layers)
    input_size = field_size*embedding_size
    glorot = np.sqrt(2.0/(input_size + deep_layers[0]))
    weights["layer_0"] = tf.Variable(
        np.random.normal(loc = 0,scale = glorot,size=(input_size,deep_layers[0])),
        dtype=tf.float32
    )
    weights["bias_0"] = tf.Variable(
        np.random.normal(loc = 0,scale=glorot,size=(1,deep_layers[0])),
        dtype=tf.float32
    )
    for i in range(1,num_layers):
        glorot = np.sqrt(2.0/(deep_layers[i-1] + deep_layers[i]))   
        weights["layer_%d"%i] = tf.Variable(
            np.random.normal(loc = 0,scale = glorot,size=(deep_layers[i-1],deep_layers[i])),
                            dtype=tf.float32)
        weights["bias_%d"%i] = tf.Variable(
            np.random.normal(loc = 0,scale = glorot,size=(1,deep_layers[i])),
                            dtype=tf.float32)
    # concatenation layer
    concate_size = field_size + embedding_size + deep_layers[-1]
    glorot_c = np.sqrt(2.0/(concate_size+1))
    weights["concate_layer"] = tf.Variable(
        np.random.normal(loc=0,scale=glorot_c,size=(concate_size,1)),
                         dtype=tf.float32
                        )
    weights["concate_bias"] = tf.Variable(
        tf.constant(0.01),dtype=tf.float32
    )
    return weights

def deepFM(dropout=None,activation_function=tf.nn.relu,batch_norm=False,feature_value=Xv,feature_index=Xi):
    weights = init_weight(deep_layers,feature_size,embedding_size)

    feature_value = tf.cast(feature_value,tf.float32)
    feature_index = tf.cast(feature_index,tf.int32)

    # (2) embedding
    #feature_index (1488028, 37)
    feature_embedding = tf.nn.embedding_lookup(weights["feature_embedding"],feature_index) # None*Field_size*K # (1488028, 37, 8)
    feature_value = tf.reshape(feature_value,shape=[-1,37,1])

    feature_embedding = tf.multiply(feature_embedding,feature_value)

    # FM component
    ## first order
    feature_bias = tf.nn.embedding_lookup(weights["feature_bias"],feature_index)
    first_order = tf.multiply(feature_bias,feature_value)
    first_order = tf.reduce_sum(first_order,axis=2)
        # first_order (1488028, 37, 1)

    ## second order
    # sum_squre
    sum_squre = tf.reduce_sum(feature_embedding,axis=1)
    sum_squre = tf.multiply(sum_squre,sum_squre)

    # squre_sum
    squre_sum = tf.multiply(feature_embedding,feature_embedding)
    squre_sum = tf.reduce_sum(feature_embedding,axis=1)
   
    second_order = 0.5*(tf.subtract(sum_squre,squre_sum))
        # second_order (1488028, 257)

    # deep component
    y_deep = tf.reshape(feature_embedding,shape=[-1,field_size*embedding_size])
    y_deep = tf.nn.dropout(y_deep,dropout[0])
    for i in range(len(deep_layers)):
        print("y_deep",y_deep.shape)
        print("layer_%d"%i,weights["layer_%d"%i].shape)
        print("bias_%d"%i,weights["bias_%d"%i])
        y_deep = tf.add(tf.matmul(y_deep,weights["layer_%d"%i]),weights["bias_%d"%i])
        # if(batch_norm):
        #     y_deep = tf.nn.batch_norm_layer()
        y_deep = tf.nn.dropout(y_deep,dropout[i])
        y_deep = activation_function(y_deep)
    # print("y_deep.shape",y_deep.shape) #(1488028, 32)

    # concatenate layer
    y_concate  = tf.concat([first_order,second_order,y_deep],axis=1) #-----
    # print("y_concate_shape",y_concate.shape)
    # print(y_concate)
    y = tf.add(tf.matmul(y_concate,weights["concate_layer"]),weights["concate_bias"])
    print(y)

    return y


# def batch_norm_layer():


# 分层交叉验证-确保数据平衡性
folds = StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train,y_train)

# parameter 
batch_size =  1024
num_epochs =  20
lr = 0.001
dropout = [0.8,0.8,0.8]
loss_type="logloss"
deep_layers = [32,32]
embedding_size = 8
optimizer_type="adam"

_get = lambda x, l: [x[i] for i in l]
def shuffle_in_union(Xi_train_, Xv_train_, y_train_):
     rds = np.random.get_state()
     np.random.shuffle(Xi_train_)
     np.random.set_state(rds)
     np.random.shuffle(Xv_train_)
     np.random.set_state(rds)
     np.random.shuffle(y_train)
     return Xi_train_,Xv_train_,y_train_

def loss(loss_type,out,label):
    label = tf.convert_to_tensor(label)
    label = tf.expand_dims(label,axis=1)

    if loss_type == "logloss":
        out = tf.nn.sigmoid(out)
        loss = tf.losses.log_loss(label, out)

    elif loss_type == "mse":
        loss = tf.nn.l2_loss(tf.subtract(label, out))
    return loss


def get_batch(Xi,Xv,y,index):
    start = index * batch_size
    end = (index+1) * batch_size
    end=end if end<len(y) else len(y)
    return Xi[start:end],Xv[start:end],y[start:end]

# optimizer
def get_optimizer(optimizer_type,lr,loss):
    if optimizer_type == "adam":
        return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8).minimize(loss)
    elif optimizer_type == "adagrad":
        return tf.train.AdagradOptimizer(learning_rate=lr,
                                                    initial_accumulator_value=1e-8).minimize(loss)
    elif optimizer_type == "gd":
        return tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    elif optimizer_type == "momentum":
        return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.95).minimize(
            loss)
    elif optimizer_type == "yellowfin":
        return YFOptimizer(learning_rate=lr, momentum=0.0).minimize(
            loss)

def train_validate_model(batch_size,num_epochs,lr,optimizer_type,loss_type="logloss"):
    # train deepFM
    # optimizer = get_optimizer(optimizer_type,lr)
    for i,(train_index, val_index) in enumerate(folds):
        # extract train_data validation_data
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_index), _get(Xv_train, train_index), _get(y_train, train_index)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, val_index), _get(Xv_train, val_index), _get(y_train, val_index)
      
        train_num = len(train_index)
        batch_num = train_num//batch_size
        trainloss = []
        valiloss = []

        for epoch in range(num_epochs):
            # 训练模型
            Xi_train_, Xv_train_, y_train_ = shuffle_in_union(Xi_train_, Xv_train_, y_train_)
            train_loss_epoch = []

            for batch in range(batch_num):
                Xi_batch, Xv_batch, y_batch = get_batch(Xi_train_, Xv_train_, y_train_,batch)
                out = deepFM(dropout, activation_function=tf.nn.relu, batch_norm=False, feature_value=Xv_batch, feature_index=Xi_batch)
            
                train_loss_batch = loss(loss_type, out, y_batch)
                train_loss_epoch.append(train_loss_batch)
                
                get_optimizer(optimizer_type,lr,loss=train_loss_batch)
                
                # # 反向传播 
                # train_loss_batch.backward()
                # # 更新参数
                # optimizer.step()

            train_loss_epoch_average = sum(train_loss_epoch) / batch_num
            trainloss.append(train_loss_epoch_average)

            # 计算验证损失
            out_val = deepFM(dropout, activation_function=tf.nn.relu, batch_norm=False, feature_value=Xv_valid_, feature_index=Xi_valid_)
            val_loss = loss(loss_type, out_val, y_valid_)
            valiloss.append(val_loss)

        print('训练损失:',i, trainloss)
        print('验证损失:',i,valiloss)    
    return trainloss,valiloss                  

train_validate_model(batch_size,num_epochs,lr,optimizer_type,loss_type)

def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()
# use train_loss_batch update our parameter
# vali_loss_epoch evaluate our model





 


            




    # def train(Xi_train,Xv_train,y_train,Xi_valid,Xv_valid,y_valid):




        
        


    # question
    # 1.what's the difference between np.random.normal and tf.random_normal
    # why we use the former in nn,the latter in embedding