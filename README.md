# food_recognizer


# 菜品识别（Fod Recognizer）Demo

### 1. 知识储备

1. 爬虫的基本思想
2. python图像库对图片进行处理
3. python二进制文件的制作
4. 深度学习中神经网络的基本理念
5. CNN（卷积神经网络）的构建

### 2. 获取数据集

#### 2-1. 爬虫获取图片

	图片源选择的是百度图片，由于百度图片是动态js加载的action，所以使用python的request库添加参数动态获取页面。爬虫代码如下：

    #将搜索关键词作为keyword参数传递，抓取想要数目的页面图片
    def getIntPages(keyword, pages):
        params = []
        for i in range(30, 30*pages+30, 30):
            params.append({
                'tn':'resultjson_com',
                'ipn': 'rj',
                'ct':'201326592',
                'is': '',
                'fp': 'result',
                'queryWord': keyword,
                'cl': '2',
                'lm': '-1',
                'ie': 'utf-8',
                'oe': 'utf-8',
                'st': '-1',
                'ic': '0',
                'word': keyword,
                'face': '0',
                'istype': '2',
                'nc': '1',
                'pn': i,
                'rn': '30'
            })
        url = 'https://image.baidu.com/search/acjson' #url 链接
        urls = []
        for i in params:
            content = requests.get(url, params=i).text
            img_urls = re.findall(r'"thumbURL":"(.*?)"', content) #正则获取图片链接
            urls.append(img_urls)
            #urls.append(requests.get(url,params = i).json().get('data'))
            #print("%d times : " % x, img_urls)
        return urls
    
    #将获取的图片保存在本地文件夹中
    def fetch_img(path,dataList):
        if not os.path.exists(path):
            os.mkdir(path)
    
        x = 0
        for list in dataList:
            for i in list:
                print("=====downloading %d/1200=====" % (x + 1))
                ir = requests.get(i)
                open(path + '%d.jpg' % x, 'wb').write(ir.content)
                x += 1
    
    
    
    if __name__ == '__main__':
        url = 'https://image.baidu.com/search/acjson'
    
        dataList = getIntPages('猪肉', 40) #获取40页猪肉的图片
        fetch_img("data/train/meat/", dataList)



#### 2-2. 图片预处理

- 图片删选
  由于百度图片中爬取的图片五花八门非常复杂，有些图片和关键词差别很大，需要手动对其进行挑选，尽可能使得数据集贴近用户拍照菜品，由于是从demo开始，只使用蔬菜和水果作为数据集，后期可以再添加分类。
- 图片压缩与转换，图片格式化（特征与标签的表示方式）
  因使用的是CNN识别cifar10的数据集，需要将图片转化为32*32 size大小的图片，并全部转化为RGB模式的图片：
      #read file then return a (1,3072) array
          def read_file(self, filename):
              img = Image.open(filename)
              img.convert('RGB') #转化为RGB
              #print(filename)
              img = img.resize((32,32)) #压缩为32*32
              try:
                  red, green, blue = img.split()
                  red_arr = pltimg.pil_to_array(red)
                  green_arr = pltimg.pil_to_array(green)
                  blue_arr = pltimg.pil_to_array(blue)
      
                  r_arr = red_arr.reshape(1024)
                  g_arr = green_arr.reshape(1024)
                  b_arr = blue_arr.reshape(1024)
      
                  result = np.concatenate((r_arr, g_arr, b_arr))
                  return result
              except (ValueError):
                  print(filename)
                  #img.show()
  压缩与转化完成后，将其与label合并后以字典的形式存储并写入文件：
          def save_pickle(self, result, label, label_name):
              print("=====saving picture, please wait=====")
      
              dic = {'label': label, 'data':result, 'label_name': label_name}
              file_path = "data/train_file/" + "data_batch_test"
              with open(file_path,'wb') as f:
                  p.dump(dic, f)
      
              print("=====save mode end=====")
  其中label是在读取文件夹的时候以文件夹名称划分：
      def get_file_name(local_path):
          label = []
          label_name = []
          file = []
          for i, dirs in enumerate(os.listdir(local_path)):
              label_name.append(dirs)
              for f in os.listdir((os.path.join(local_path,dirs))):
                  label.append(i)
                  img_path = os.path.join(os.path.join(local_path,dirs), f)
                  file.append(img_path)
          return file, label, label_name
  至此图片预处理部分完毕

#### 2-3. 训练、测试数据的准备

- 使用上述代码将处理好的2692张蔬菜水果图片制作为训练数据集data_batch_train，将另外299张图片制作为测试数据集data_batch_test
- 读取到data_batch文件中的data与label并进行存储返回
      #从文件中读取所有数据【label，image_data】
      def unpickle(filename):
          with open(filename, 'rb') as f:
              dict = p.load(f, encoding='bytes')
          return dict
      
      #分离文件中的label和data
      def load_data_once(filename):
          batch = unpickle(filename)
          data = batch['data']
          labels = batch['label']
          print("reading data and labels from %s" % filename)
          return data,labels
      
      def load_data(filequeue, data_dir, labels_count):
          global image_size, image_channels
      
          data, labels = load_data_once(data_dir + '/' + filequeue[0])
          for f in filequeue[1:]:
              data_f, label_f = load_data_once(data_dir + '/' + f)
              data = np.append(data,data_f,axis=0)
              labels = np.append(labels, label_f,axis = 0)
          labels = np.array([ [float(i == label) for i in range(labels_count) ]
                              for label in labels])
          data = data.reshape([-1,image_channels, image_size, image_size])
          data = data.transpose([0,2,3,1])
          return data, labels
  由于训练数据集不是很多，选择对图像进行截取，移动等方法再进一步处理：
      #图片截取
      def random_crop(batch, crop_shape, padding = None):
          img_shape = np.shape(batch[0])
      
          if padding:
              img_shape = (img_shape[0] + 2*padding,img_shape[1], 2*padding)
          new_batch=[]
          newPad = ((padding,padding), (padding,padding), (0,0))
          for i in range(len(batch)):
              new_batch.append(batch[i])
              if padding:
                  new_batch[i] = np.lib.pad(batch[i], pad_width=newPad,
                                            mode='constant', constant_values=0)
                  new_height = random.randint(0, img_shape[0] - crop_shape[0])
                  new_wight= random.randint(0, img_shape[1] - crop_shape[1])
                  new_batch[i] = new_batch[i][new_height:new_height + crop_shape[0],
                                 new_wight:new_wight + crop_shape[1]]
          return new_batch
      
      #图片数组左右移动
      def random_flip_leftRight(batch):
          for i in range(len(batch)):
              if bool(random.getrandbits):
                  batch[i] = np.fliplr(batch[i])
          return batch
      
      #RGB数组预处理
      def color_preProcess(x_train, x_test):
          x_train = x_train.astype('float32')
          x_test = x_test.astype('float32')
          x_train[:,:,:,0] = (x_train[:,:,:,0] - np.mean(x_train[:,:,:,0])) / np.std(x_train[:,:,:,0])
          x_train[:,:,:,1] = (x_train[:,:,:,1] - np.mean(x_train[:,:,:,1])) / np.std(x_train[:,:,:,1])
          x_train[:,:,:,2] = (x_train[:,:,:,2] - np.mean(x_train[:,:,:,2])) / np.std(x_train[:,:,:,2])
      
          x_test[:,:,:,0] = (x_test[:,:,:,0] - np.mean(x_test[:,:,:,0])) / np.std(x_test[:,:,:,0])
          x_test[:,:,:,1] = (x_test[:,:,:,1] - np.mean(x_test[:,:,:,1])) / np.std(x_test[:,:,:,1])
          x_test[:,:,:,2] = (x_test[:,:,:,2] - np.mean(x_test[:,:,:,2])) / np.std(x_test[:,:,:,2])
      
          return x_train, x_test
      
      #返回新的图片数组batch
      def data_augmentation(batch):
          batch = random_flip_leftRight(batch)
          batch = random_crop(batch, [32,32], 4)
          return batch
  

### 3. 算法的选择与模型训练

#### 3-1. 算法选择

	使用广为流传的识别cifar10数据集的CNN神经网络，原因有二，一是CNN 作为比较成熟的算法模型，在cifar10分类中可以获得91%左右的准确率，在mnist数据集中可以获得97%以上的的准确率；二是CNN模型的构建相对容易理解，属于比较基础的模型之一，方便以后的调参优化改进。

#### 3-2. 模型搭建

首先定义一个conv函数创建卷积层：

    def conv(x, is_train, shape):
        he_initializer = tf.contrib.keras.initializers.he_normal()
        W = tf.get_variable('weights', shape=shape, initializer=he_initializer)
        b = tf.get_variable('bias', shape=[shape[3]], initializer=tf.zeros_initializer)
        x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=is_train,
                                            updates_collections=None)



模型使用三层卷积，命名为conv1, conv2, conv3, 每层卷积后创建2层池化层， 命名为mlp1-1， mlp1-2，再进行最大池化、dropout防止过拟合后返回一层输出层并作为下一层的输入。最后第三层卷积后的输出作为softmax层的输入，返回图像数据与label数组。

    with tf.variable_scope('conv1'):
        output = conv(x, use_bn, [5, 5, 3, 192])
        output = activation(output)
    
    with tf.variable_scope('mlp1-1'):
        output = conv(output, use_bn, [1, 1, 192, 160])
        output = activation(output)
    
    with tf.variable_scope('mlp1-2'):
        output = conv(output, use_bn, [1, 1, 160, 96])
        output = activation(output)
    
    with tf.name_scope('max_pool-1'):
        output = max_pool(output, 3, 2)
    
    with tf.name_scope('dropout-1'):
        output = tf.nn.dropout(output, keep_prob)
    
    with tf.variable_scope('conv2'):
        output = conv(output, use_bn, [5, 5, 96, 192])
        output = activation(output)
    
    with tf.variable_scope('mlp2-1'):
        output = conv(output, use_bn, [1, 1, 192, 192])
        output = activation(output)
    
    with tf.variable_scope('mlp2-2'):
        output = conv(output, use_bn, [1, 1, 192, 192])
        output = activation(output)
    
    with tf.name_scope('max_pool-2'):
        output = max_pool(output, 3, 2)
    
    with tf.name_scope('dropout-2'):
        output = tf.nn.dropout(output, keep_prob)
    
    with tf.variable_scope('conv3'):
        output = conv(output, use_bn, [3, 3, 192, 192])
        output = activation(output)
    
    with tf.variable_scope('mlp3-1'):
        output = conv(output, use_bn, [1, 1, 192, 192])
        output = activation(output)
    
    with tf.variable_scope('mlp3-2'):
        output = conv(output, use_bn, [1, 1, 192, 2])
        output = activation(output)
    
    with tf.name_scope('global_avg_pool'):
        output = global_avg_pool(output, 8, 1)
    
    with tf.name_scope('softmax'):
        output = tf.reshape(output, [-1, 2]) #由于当前只有两个分类，故softmax输出一个两列数组

模型创建好后将交叉熵，损失精度，训练步数和预测等变量加入tensor：

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    
        with tf.name_scope('l2_loss'):
            l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    	
        #使用Momentum 作为迭代优化器
        with tf.name_scope('train_step'):
            train_step = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum, use_nesterov=True).minimize(
                cross_entropy + l2 * FLAGS.weight_decay)
    
        with tf.name_scope('prediction'):
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

然后定义saver变量作为Saver函数的调用保存model：

    saver = tf.train.Saver()

#### 3-3. 模型的训练与评估

模型搭建完毕后创建一个session对模型进行训练和测试，使用flag变量作为参数，当迭代次数一定后保存模型并输出当前模型的预测精度：

    sess.run(tf.global_variables_initializer())
            #saver.restore(sess, "./check/model.ckpt")
            summary_writer = tf.summary.FileWriter(FLAGS.log_save_path, sess.graph)
    
            for ep in range(1, FLAGS.epochs + 1):
                lr = learning_rate_schedule(ep)
                pre_index = 0
                train_acc = 0.0
                train_loss = 0.0
                start_time = time.time()
    
                print("\nepoch %d/%d:" % (ep, FLAGS.epochs))
    
                for it in range(1, FLAGS.iteration + 1):
                    if pre_index + FLAGS.batch_size < 50000: #分批次提取训练数据
                        batch_x = train_x[pre_index:pre_index + FLAGS.batch_size]
                        batch_y = train_y[pre_index:pre_index + FLAGS.batch_size]
                    else:
                        batch_x = train_x[pre_index:]
                        batch_y = train_y[pre_index:]
    
                    batch_x = data_augmentation(batch_x)
    
                    _, batch_loss = sess.run([train_step, cross_entropy],
                                             feed_dict={x: batch_x, y_: batch_y, use_bn: True, keep_prob: FLAGS.dropout,
                                                        learning_rate: lr})
                    batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, use_bn: True, keep_prob: 1.0}) #精度预测
    
                    train_loss += batch_loss
                    train_acc += batch_acc
                    pre_index += FLAGS.batch_size
    				
                    #测试数据集的评估
                    if it == FLAGS.iteration:
                        train_loss /= FLAGS.iteration
                        train_acc /= FLAGS.iteration
    
                        train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                                                          tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])
    
                        val_acc, val_loss, test_summary = run_testing(sess)
    
                        summary_writer.add_summary(train_summary, ep)
                        summary_writer.add_summary(test_summary, ep)
                        summary_writer.flush()
    
                        print(
                            "iteration: %d/%d, cost_time: %ds, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f" % (
                            it, FLAGS.iteration, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
                        #checkpt_path = "./check/model.ckpt"
                        #saver.save(sess, checkpt_path)
                        #print("Model saved in file: %s" % save_path)
                    else: #训练数据集的评估
                        print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f" % (
                        it, FLAGS.iteration, train_loss / it, train_acc / it), end='\r')
                        #checkpt_path = "./check/model.ckpt"
                        #saver.save(sess, checkpt_path)

在经过82次模型的运行，使用0.01的learning rate， 每次进行43次迭代后可以看到该模型预测精度可以达到97%：



### 4. 新样本预测

- 图片转换、预处理
  当使用模型预测一个新的样本的时候，同样需要将图片转化为32*32的size并转化为RGB模式的图片以防图片识别失败，之所以要加转为RGB式语句是因为在预处理中遇到过valueError，提示在RGB三通道split的时候返回值不匹配，可能是图片本身并不是RGB模式导致的。
- 特征格式化表示
  进行预处理完毕后需要将该图片转为一个1*3072的array，其中第一个1024维度为Red通道，第二个1024维度为Green通道，第三个1024维度为Blue通道
  至此图片特征格式化完毕。

### 5. 后期优化

由于此算法模型只是一个demo，后续还需要一些优化，有以下几个方面：

- 对图片的删选，在制作训练数据集的时候还需要进一步对图片进行处理，挑选一些和用户拍照接近的菜品
- 需要更多的菜品分类来更贴近生活如添加肉类分类流程工作图：

### 6. 模型流程图


