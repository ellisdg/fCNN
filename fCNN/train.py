
# coding: utf-8

# In[1]:


import sys
sys.path.append("/home/neuro-user/Data/REPORTS/Radiomics/")


# In[2]:


import pandas as pd
import os
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from resnet import load_model, ResnetBuilder, sensitivity, specificity, j_stat
import numpy as np
from utils import load_json
from radiomic_utils import SingleSiteSequence, load_fs_lut


# # 1. Load configuration data

# In[3]:


config_file = '/home/neuro-user/Data/REPORTS/Radiomics/12B_multiscanner_dti_training_config.json'
resnet_config = load_json(config_file)
training_files = resnet_config['training_files']['HCP']
validation_files = resnet_config['validation_files']['HCP']
window = np.asarray(resnet_config['window'])
spacing = np.asarray(resnet_config['spacing'])
input_shape = resnet_config['input_shape']


# # 2. Create model

# In[4]:


model_path = '../01_data/12_hcp_language_dti_model_resnet34.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    input_shape = tuple(window.tolist() + [4])
    model = ResnetBuilder.build_resnet_34(input_shape, 1, activation='sigmoid')
    model.compile(metrics=['accuracy', sensitivity, specificity, j_stat], 
                  optimizer='Adam', loss='binary_crossentropy')


# # 4. Create Generators

# In[4]:


batch_size = 50
points_per_subject = 10
validation_points_per_subject = 75
validation_batch_size = 75


# In[5]:


fs_lut = load_fs_lut()
label_names = ["parsopercularis",
               "parstriangularis",
               "superiortemporal",
               "supramarginal"]
target_labels = list()
for hemi in ('lh', 'rh'):
    for label_name in label_names:
        target_labels.append(fs_lut['ctx-{}-{}'.format(hemi, label_name)])
target_labels = tuple(target_labels)


# In[7]:


training_generator = SingleSiteSequence(filenames=training_files,
                                        batch_size=batch_size,
                                        flip=False,
                                        reorder=False,
                                        window=window,
                                        target_labels=target_labels,
                                        spacing=spacing,
                                        points_per_subject=points_per_subject)

validation_generator = SingleSiteSequence(filenames=validation_files,
                                          batch_size=validation_batch_size,
                                          flip=False,
                                          reorder=False,
                                          window=window,
                                          target_labels=target_labels,
                                          spacing=spacing,
                                          points_per_subject=validation_points_per_subject)


# # 5. Run Training

# In[32]:


checkpointer = ModelCheckpoint(filepath=model_path,
                               verbose=1,
                               save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.5,
                              patience=20, 
                              min_lr=1e-5)
csv_logger = CSVLogger('../01_data/12_hcp_dti_freesurfer-labels_training_log.csv',
                       append=True)
history = model.fit_generator(generator=training_generator, 
                              epochs=2000,
                              use_multiprocessing=True,
                              workers=10,
                              max_queue_size=20,
                              callbacks=[checkpointer, reduce_lr, csv_logger],
                              validation_data=validation_generator)


# # 6. Save configuration

# In[8]:


config = {"window": window.tolist(),
          "spacing": spacing.tolist(),
          "input_shape": input_shape,
          "target_labels": target_labels,
          "validation_files": validation_files,
          "training_files": training_files}

from utils import dump_json

dump_json(config, "../01_data/12_dti_hcp_freesurfer-labels_config.json")

