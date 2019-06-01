BATCH_SIZE = 4
EDGE_CROP = 16
NB_EPOCHS = 5
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
NET_SCALING = None
IMG_SCALING = (1, 1)
VALID_IMG_COUNT = 400
MAX_TRAIN_STEPS = 500
AUGMENT_BRIGHTNESS = False

import os
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from skimage.util import montage
montage_rgb = lambda x:np.stack([montage(x[:,:,:,i]) for i in range(x.shape[3])], -1) #将图片以蒙太奇的方式拼接起来

train_image_dir = '../airbus-ship-detection/train_v2/'
test_image_dir = '../airbus-ship-detection/test_v2/'
import gc; gc.enable()
from skimage.morphology import label #区域划分

# 重复编码，前面经过一次rle0,现在再来一次rle1
def multi_rle_encode(img):
    labels = label(img[:, :, 0])  #区域划分标记
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# rle 编码
def rle_encode(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_le, shape=(768, 768)):
    s = mask_le.split()
    starts, lengths = [np.array(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  #对齐rle方向

# 将mask以图片的形式来展示出来
def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype=np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks +=rle_decode(mask)
    return np.expand_dims(all_masks, -1)

# 读入数据
masks = pd.read_csv('../airbus-ship-detection/' + 'train_ship_segmentations_v2.csv')
print(masks.shape)
print(masks.head())
print(masks['ImageId'].value_counts().shape[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']
img_0 = masks_as_image(rle_0) #转成图像来显示mask
ax1.imshow(img_0[:, :, 0])
ax1.set_title('Image0')
# plt.show()

rle_1 = multi_rle_encode(img_0)
img_1 = masks_as_image(rle_1) #转成图像来显示mask
ax2.imshow(img_1[:, :, 0])
ax2.set_title('Image1')
plt.show()
#打印编码长度的变化
print('rle_0:', len(rle_0), '\n', 'rle_1', len(rle_1))


# 添加部分特征
masks['ships'] = masks['EncodedPixels'].map(lambda c_crow:1 if isinstance(c_crow, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: os.stat(os.path.join(train_image_dir, c_img_id)).st_size/1024)
unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb']>50] #只保存50kb的文件
unique_img_ids['file_size_kb'].hist()
plt.show()
masks.drop(['ships'], axis=1, inplace=True)
unique_img_ids.sample(5)
#划分数据集
from sklearn.model_selection import train_test_split
train_ids, valid_ids = train_test_split(unique_img_ids, test_size=0.3, stratify=unique_img_ids['ships'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0])
print(valid_df.shape[0])

train_df['ships'].hist()
plt.show()

train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x+1)//2).clip(0, 7)
#采样,对没有船的图片进行更小的采样
def sample_ships(in_df, base_rep_val=1500):
    if in_df['ships'].values[0] ==0:
        return in_df.sample(base_rep_val//3) #如果没有船则随机选取500行数据
    else:
        return in_df.sample(base_rep_val, replace=(in_df.shape[0]<base_rep_val)) #如果少就直接以代替的方式补充数据

balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)
balanced_train_df['ships'].hist(bins=np.arange(10))

# 读进数据
# 数据生成器
def make_image_gen(in_df, batch_size=BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = train_image_dir + c_img_id
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []

# 生成训练数据
train_gen = make_image_gen(balanced_train_df)
train_x, train_y = next(train_gen)
print(train_x.shape, train_x.min(), train_y.max())
print(train_y.shape, train_y.min(), train_y.max())

# 展示拼接效果，从左到右是原始拼接图，分割拼接图，边缘标记图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
batch_rgb = montage_rgb(train_x)
batch_seg = montage(train_y[:, :, :, 0])
ax1.imshow(batch_rgb)
ax1.set_title('Images')
ax2.imshow(batch_seg)
ax2.set_title('Segmentation')
ax3.imshow(mark_boundaries(batch_rgb, batch_seg.astype(int)))
ax3.set_title('Outlined Ships')
plt.show()

# 生成验证数据
valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))
print(valid_x.shape, valid_y.shape)

from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(
    rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.01,
    zoom_range = [0.9, 1.25],
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'reflect',
    data_format = 'channels_last'
)

if AUGMENT_BRIGHTNESS:
    dg_args['brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

if AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)

# 生成增强数据集
def create_aug_gen(in_gen, seed=None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        g_x = image_gen.flow(255*in_x,
                             batch_size=in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        yield next(g_x)/255.0, next(g_y)


# 显示增强后的数据
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)
t_x = t_x[:9]
t_y = t_y[:9]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(montage_rgb(t_x), cmap='gray')
ax1.set_title('images')
ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray_r')
ax2.set_title('ships')
plt.show()

gc.collect()

# 构建模型
from keras import models, layers
def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)

if UPSAMPLE_MODE == 'DECONV':
    upsample = upsample_conv
else:
    upsample = upsample_simple

input_img = layers.Input(t_x.shape[1:], name='RGB_Input')
pp_in_layer = input_img
if NET_SCALING is not None:
    pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)
pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
pp_in_layer = layers.BatchNormalization()(pp_in_layer)

c1 = layers.Conv2D(8, (3,3), activation='relu', padding='same')(pp_in_layer)
c1 = layers.Conv2D(8, (3,3), activation='relu', padding='same')(c1)
p1 = layers.MaxPooling2D((2,2))(c1)

c2 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(p1)
c2 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(c2)
p2 = layers.MaxPooling2D((2,2))(c2)

c3 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(p2)
c3 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c3)
p3 = layers.MaxPooling2D((2,2))(c3)

c4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p3)
c4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c4)
p4 = layers.MaxPooling2D((2,2))(c4)

c5 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p4)
c5 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c5)

u6 = upsample(64, (2,2), strides=(2,2), padding='same')(c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u6)
c6 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c6)

u7 = upsample(32, (2,2), strides=(2,2), padding='same')(c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u7)
c7 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c7)

u8 = upsample(16, (2,2), strides=(2,2), padding='same')(c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(u8)
c8 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(c8)

u9 = upsample(8, (2,2), strides=(2,2), padding='same')(c8)
u9 = layers.concatenate([u9, c1])
c9 = layers.Conv2D(8, (3,3), activation='relu', padding='same')(u9)
c9 = layers.Conv2D(8, (3,3), activation='relu', padding='same')(c9)

d = layers.Conv2D(1, (1,1), activation='sigmoid')(c9)
d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
d = layers.ZeroPadding2D((EDGE_CROP,EDGE_CROP))(d)
if NET_SCALING is not None:
    d = layers.UpSampling2D(NET_SCALING)(d)

seg_model = models.Model(inputs=[input_img], outputs=[d])
seg_model.summary()

import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

# dice系数
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union =  K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

# binary cross entropy + dice loss
def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred))) / K.sum(y_true)

seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate] )

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
weight_path = "{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=3, verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor='val_dice_coef', mode='max', patience=15)
tesorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images=True, write_grads=True)
callbacks_list = [checkpoint, early, reduceLROnPlat, tesorboard]

step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0]//BATCH_SIZE)
aug_gen = create_aug_gen(make_image_gen(balanced_train_df))
loss_history = [seg_model.fit_generator(
    aug_gen,
    steps_per_epoch=step_count,
    epochs=NB_EPOCHS,
    validation_data=(valid_x, valid_y),
    callbacks = callbacks_list,
    workers=1
)]


def show_loss(loss_history):
    epich = np.cumsum(np.concatenate([np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich,
                 np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich,
                 np.concatenate([mh.history['true_positive_rate'] for mh in loss_history]),
                 'b-',
                 epich,
                 np.concatenate([mh.history['val_true_positive_rate'] for mh in loss_history]),
                 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('True Positive Rate\n(Positive Accuracy)')

    _ = ax3.plot(epich,
                 np.concatenate([mh.history['binary_accuracy'] for mh in loss_history]),
                 'b-',
                 epich,
                 np.concatenate([mh.history['val_binary_accuracy'] for mh in loss_history]),
                 'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('Binary Accuracy (%)')

    _ = ax4.plot(epich,
                 np.concatenate([mh.history['dice_coef'] for mh in loss_history]),
                 'b-',
                 epich,
                 np.concatenate([mh.history['val_dice_coef'] for mh in loss_history]),
                 'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('DICE')
    plt.show()


show_loss(loss_history)


seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')

pred_y = seg_model.predict(valid_x)
print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
ax.hist(pred_y.ravel(), np.linspace(0, 1, 10))
ax.set_xlim(0, 1)
ax.set_yscale('log', nonposy='clip')
plt.show()

# 测试数据
test_paths = os.listdir(test_image_dir)
print(len(test_paths), 'test images found')


if IMG_SCALING is not None:
    fullres_model = models.Sequential()
    fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
    fullres_model.add(seg_model)
    fullres_model.add(layers.UpSampling2D(IMG_SCALING))
else:
    fullres_model = seg_model
fullres_model.save('fullres_model.h5')


fig, m_axs = plt.subplots(20, 2, figsize = (10, 40))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = imread(c_path)
    first_img = np.expand_dims(c_img, 0)/255.0
    first_seg = fullres_model.predict(first_img)
    ax1.imshow(first_img[0])
    ax1.set_title('Image')
    ax2.imshow(first_seg[0, :, :, 0], vmin = 0, vmax = 1)
    ax2.set_title('Prediction')
    plt.show()
fig.savefig('test_predictions.png')

























