import tensorflow as tf
import numpy as np
import scipy
import scipy.io, scipy.misc
import skimage.io, skimage.data, skimage.filters
import matplotlib.pyplot as plt

VGG_PATH = "data/vgg/imagenet-vgg-verydeep-19.mat"
STYLE_PATH = "data/style/la_muse.jpg"
CONTENT_PATH = "data/img/father2.jpg"
SAVE_PATH = "data/output/"
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))

STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
STYLE_WEIGHT = [0.2, 0.2, 0.2, 0.2, 0.2]
CONTENT_LAYER = 'conv4_2'
MIX_RATIO = 0.3
SCALE = 0.1

style_ratio = 1000
content_ratio = 1.
tv_ratio = 10.
iteration = 5001
learning_rate = 5e-3


def vgg_net(data_path, input_image):
    '''
    load the VGG-19 Net, which used to evaluate the images' similarity of content and style.
    '''
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(data_path)
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]     # acquire the layer's type such as 'conv', 'relu' or 'pool'
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))   # exchange the position of first two params
            bias = bias.reshape(-1)
            current = tf.nn.conv2d(current, tf.constant(kernels),
                                   strides=[1,1,1,1], padding='SAME', name=name)
            current = tf.nn.bias_add(current, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = tf.nn.max_pool(current, ksize=[1,2,2,1], strides=[1,2,2,1],
                                     padding='SAME', name = name)
        net[name] = current
    return net


def load_img(src, img_size=False, norm=True, filter=True):
    img = skimage.io.imread(src)
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    if filter:
        img = img_filter(img)
    if norm:
        img = (img - MEAN_VALUES) / 255
    img = img.astype('float32')
    # print('have loaded image:%s'%src)
    return img


def save_img(path, img, img_size=False, norm=True, filter=True):
    if len(img.shape) == 4 and img.shape[0] == 1:
        img = np.reshape(img, [img.shape[1], img.shape[2], 3])
    if norm:
        img = img * 255 + MEAN_VALUES
        img = np.clip(img, 0, 255)  # 防止像素上溢出和下溢
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    img = img.astype('uint8')
    if filter:
        img = img_filter(img)
    # print('have save image:%s'%src)
    skimage.io.imsave(path, img)


def img_filter(img):
    # remove the noise by 'median' method
    for i in range(img.shape[2]):
        img[:, :, i] = skimage.filters.median(img[:, :, i])
    return img


def save_layer_img(img, path=SAVE_PATH, name='conv'):
    # print(img.shape)
    if len(img.shape) == 3:
        plt.imsave(path+name, img)
    elif len(img.shape) == 4:
        height, width = img.shape[1], img.shape[2]
        _img = np.transpose(img, [3, 1, 2, 0])
        for i in range(_img.shape[0]):
            sub_img = _img[i, :, :]
            sub_img = np.reshape(sub_img, (height, width))
            sub_img = sub_img * 255 + 128
            sub_img = sub_img.astype('uint8')
            plt.imsave(path + name + str(i), sub_img)


def get_noise_image(img, mix_ratio=0.5, loc=0.0, scale=0.3, norm=False, img_size=False):
    '''
    img: image with type 'numpy.array'.
    mix_ratio: the proportion of noise in the noise image.
        for instance:
        noise_img = img * (1 - mix_ratio) + mix_ratio * noise
    loc: locMean ("centre") of the noise distribution.
    scale: Standard deviation (spread or "width") of the noise distribution.
    norm: whether normalize the noise_image.
        if the input img was normalized before this step, this param should be False.
    '''
    height, width = img.shape[0], img.shape[1]
    noise_image = np.random.normal(loc=loc, scale=scale, size=(height, width))
    noise_image = np.array([noise_image, noise_image, noise_image]).transpose([1,2,0])
    if norm:
        noise_image = ((noise_image - MEAN_VALUES) / 255)
    if img_size != False:
        noise_image = scipy.misc.imresize(noise_image, img_size)
    noise_image = (1-mix_ratio) * img + mix_ratio * noise_image
    noise_image = noise_image.astype('float32')
    # save_img(SAVE_PATH+'noise_img.jpg', noise_image)
    return noise_image


def get_style(img, style_layers):
    '''
    evaluate the style feature by vgg_net.
    '''
    net = vgg_net(VGG_PATH, img)
    styles = []
    for layer in style_layers:
        feature = net[layer]
        _, height, width, channels = (i.value for i in feature.shape)
        size = height * width * channels
        feature = tf.reshape(feature, shape=[-1, channels])
        style = tf.matmul(tf.transpose(feature), feature) / size
        styles.append(style)
    return styles


def get_style_loss(target_style, img_style, weight):
    total_loss = 0
    num_layer = len(target_style)
    for i in range(num_layer):
        channels = target_style[i].shape[1]
        loss = tf.nn.l2_loss(target_style[i] - img_style[i]) / 4
        total_loss += loss * weight[i]
    return total_loss


def get_content(img, content_layer):
    '''
    evaluate the content feature by vgg_net.
    '''
    net = vgg_net(VGG_PATH, img)
    feature = net[content_layer]
    _, height, width, channels = (i.value for i in feature.shape)
    size = height * width * channels
    content = feature
    return content


def get_content_loss(target_content, img_content):
    _, height, width, channels = (i.value for i in img_content.shape)
    size = height * width * channels
    loss = tf.nn.l2_loss(target_content - img_content) / (4 * size)
    return loss


def get_tv_loss(img):
    """
    Compute total variation loss.

    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    batch, height, width, channels = (i.value for i in img.shape)
    h_variance = tf.nn.l2_loss((img[:, :, 1:, :] - img[:, :, :-1, :])) / height
    w_variance = tf.nn.l2_loss((img[:, 1:, :, :] - img[:, :-1, :, :])) / width

    loss = h_variance + w_variance
    return loss


def main():
    # load the images
    content_img = load_img(CONTENT_PATH, img_size=0.8)
    style_img = load_img(STYLE_PATH, img_size=1.)
    noise_img = get_noise_image(content_img, mix_ratio=MIX_RATIO, scale=SCALE)

    # transform images' shape from 3 dimensions to 4 dimensions
    content_img = tf.expand_dims(content_img, axis=0)
    style_img = tf.expand_dims(style_img, axis=0)
    noise_img = tf.expand_dims(noise_img, axis=0)
    output_img = tf.Variable(noise_img, name='output')

    # calculate the target style and target content
    target_content = get_content(content_img, CONTENT_LAYER)
    target_style = get_style(style_img, STYLE_LAYERS)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        tc, ts = sess.run([target_content, target_style])

    # loss
    output_style = get_style(output_img, STYLE_LAYERS)
    output_content = get_content(output_img, CONTENT_LAYER)
    style_loss = get_style_loss(ts, output_style, STYLE_WEIGHT)
    content_loss = get_content_loss(tc, output_content)
    tv_loss = get_tv_loss(output_img)
    loss = content_ratio * content_loss + style_ratio * style_loss + tv_ratio * tv_loss

    # optimizer
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(iteration):
            sl, cl, tvl, _loss, _ = sess.run([style_loss, content_loss, tv_loss, loss, train_step])
            if step % 500 == 0:
                print("step:%d, total_loss:%f, style_loss:%f, content_loss:%f, tv_loss:%f"
                      %(step,_loss,style_ratio*sl,content_ratio*cl,tv_ratio*tvl))
                save_img(SAVE_PATH+'output_step%d.jpg'%step, output_img.eval())


if __name__ == "__main__":
    main()