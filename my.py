import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('faces_only', True, 'If True, generates bounding boxes '
                     'for pet faces.  Otherwise generates bounding boxes (as '
                     'well as segmentations for full pet bodies).  Note that '
                     'in the latter case, the resulting files are much larger.')
FLAGS = flags.FLAGS




def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False,):
  #data
  '''
{'folder': 'OXIIIT', 'filename': 'basset_hound_103.jpg', 'source': {'database': 'OXFORD-IIIT Pet Dataset', 'annotation': 'OXIIIT', 'image': 'flickr'}, 'size': {'width': '500', 'height': '327', 'depth': '3'}, 'segmented': '0', 'object': [{'name': 'dog', 'pose': 'Frontal', 'truncated': '0', 'occluded': '0', 'bndbox': {'xmin': '244', 'ymin': '76', 'xmax': '370', 'ymax': '179'}, 'difficult': '0'}]}
  '''
    

  #label_map_dict
  #{'computer': 1, 'monitor': 2, 'scuttlebutt': 3, 'water dispenser': 4, 'drawer chest': 5}

  #image_subdirectory
  #mydata/images
  
  #获取到图片的位置
  img_path = os.path.join(image_subdirectory, data['filename'])
  #/home/chen/allTools/models-r1.5/research/petData/images/Birman_142.jpg

    
  
  with tf.gfile.GFile(img_path, 'rb') as fid:
    #读取图片
    encoded_jpg = fid.read()
  #变为2进制
  encoded_jpg_io = io.BytesIO(encoded_jpg)

  #打开图片
  image = PIL.Image.open(encoded_jpg_io)
  
  #不是jpg格式就报错
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')

  #把图片变为一个hash值
  key = hashlib.sha256(encoded_jpg).hexdigest()

    
  #获取到宽度
  width = int(data['size']['width'])
  #获取高度
  height = int(data['size']['height'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []

  for obj in data['object']:
    #获取到值
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue
    #添加到列表中
    difficult_obj.append(int(difficult))

    #获取各个坐标
    xmin = float(obj['bndbox']['xmin'])
    xmax = float(obj['bndbox']['xmax'])
    ymin = float(obj['bndbox']['ymin'])
    ymax = float(obj['bndbox']['ymax'])

    #添加坐标到列表中
    xmins.append(xmin / width)
    ymins.append(ymin / height)
    xmaxs.append(xmax / width)
    ymaxs.append(ymax / height)

    #'filename': 'basset_hound_103.jpg'
    class_name = obj['name']

    #添加文件名到列表中
    classes_text.append(class_name.encode('utf8'))

    #{'computer': 1, 'monitor': 2, 'scuttlebutt': 3, 'water dispenser': 4, 'drawer chest': 5}
    #获取到类型的参数
    classes.append(label_map_dict[class_name])
    
    #truncated': '0'
    truncated.append(int(obj['truncated']))
    
    #'pose': 'Frontal'
    poses.append(obj['pose'].encode('utf8'))
  
  
  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }
  
  #生成tfrecord
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example



def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     filenames):
  #output_filename
  #/home/chen/allTools/models-r1.5/research/petOut/pet_train_with_masks.record

  #label_map_dict
  #{'computer': 1, 'monitor': 2, 'scuttlebutt': 3, 'water dispenser': 4, 'drawer chest': 5}

  #annotations_dir
  #mydata/annotations
  
  #image_dir
  #mydata/images
  writer = tf.python_io.TFRecordWriter(output_filename)
  
  for xml_path in filenames:
       with tf.gfile.GFile(xml_path, 'r') as fid:
       	 #获取到xml
         xml_str = fid.read()
       #将xml变为树状
       xml = etree.fromstring(xml_str)
       #获取数据
       data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
       try:
          tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
	  #data

	  #{'folder': 'OXIIIT', 'filename': 'basset_hound_103.jpg', 'source': {'database': 'OXFORD-IIIT Pet Dataset', 'annotation': 'OXIIIT', 'image': 'flickr'}, 'size': {'width': '500', 'height': '327',
          #'depth': '3'}, 'segmented': '0', 'object': [{'name': 'dog', 'pose': 'Frontal', 'truncated': '0', 'occluded': '0', 'bndbox': {'xmin': '244', 'ymin': '76', 'xmax': '370', 'ymax': '179'},     
          # 'difficult': '0'}]}


	  #label_map_dict
	  #{'computer': 1, 'monitor': 2, 'scuttlebutt': 3, 'water dispenser': 4, 'drawer chest': 5}

	  #image_dir
	  #mydata/images=
          writer.write(tf_example.SerializeToString())
       except ValueError:
      	  logging.warning('Invalid example: %s, ignoring.', xml_path)
  writer.close()
  
  


def main(_):
  data_dir = FLAGS.data_dir
  #mydata/
  
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  #{'computer': 1, 'monitor': 2, 'scuttlebutt': 3, 'water dispenser': 4, 'drawer chest': 5}

  image_dir = os.path.join(data_dir, 'images')
  #mydata/images

  annotations_dir = os.path.join(data_dir, 'annotations')
  #mydata/annotations
  
  #定义所有的文件列表
  allFilesNames = []
 
  #地址这里要修改
  rootdir  = '/home/chen/allTools/models-r1.5/research/mydata/annotations/xmls'#存放图片的文件夹路径
  for parent,dirnames,filenames in os.walk(rootdir):
    for fileName in filenames:
      #获取到所有的文件路径
      allFilesNames.append(rootdir+"/"+fileName)

  random.seed(42)
  random.shuffle(allFilesNames)
  #获取到总共的数量
  allFilesNum = int(len(allFilesNames))
  
  #训练集的数量
  trainNum = int(allFilesNum*0.7)
  #获取训练集
  train_files = allFilesNames[:trainNum]
	
  
  #获取测试集
  val_files = allFilesNames[trainNum:]
  #训练集的数据
  print(int(allFilesNum*0.3))
  #46


  #获取训练集的位置
  train_output_path = os.path.join(FLAGS.output_dir, 'pet_train.record')
  #/home/chen/allTools/models-r1.5/research/petOut/pet_train.record

  #获取验证集的位置
  val_output_path = os.path.join(FLAGS.output_dir, 'pet_val.record')
  #/home/chen/allTools/models-r1.5/research/petOut/pet_val.record
 
  create_tf_record(train_output_path, label_map_dict, annotations_dir,
                   image_dir, train_files)
  #train_output_path
  #/home/chen/allTools/models-r1.5/research/petOut/pet_train_with_masks.record

  #label_map_dict
  #{'computer': 1, 'monitor': 2, 'scuttlebutt': 3, 'water dispenser': 4, 'drawer chest': 5}

  #annotations_dir
  #mydata/annotations
  
  #image_dir
  #mydata/images

  create_tf_record(val_output_path, label_map_dict, annotations_dir,
                   image_dir,val_files)
  
  #val_output_path
  #/home/chen/allTools/models-r1.5/research/petOut/pet_val.record

  #label_map_dict
  #{'computer': 1, 'monitor': 2, 'scuttlebutt': 3, 'water dispenser': 4, 'drawer chest': 5}

  #annotations_dir
  #mydata/annotations

  #image_dir
  #mydata/images

  #train_files
  #xml的文件路径



if __name__ == '__main__':
  tf.app.run()
