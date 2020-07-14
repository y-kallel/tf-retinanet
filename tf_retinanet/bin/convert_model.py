#!/usr/bin/env python

"""
Copyright 2017-2019 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import tf_retinanet.bin  # noqa: F401
	__package__ = "tf_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..              import models
from ..backbones     import get_backbone
from ..utils.anchors import parse_anchor_parameters
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.gpu     import setup_gpu
from ..utils.config  import parse_yaml, parse_additional_options


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.
    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'no_resize'        : args.no_resize,
        'preprocess_image' : preprocess_image,
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None
	if args.dataset_type == 'csv':
		train_generator = CSVGenerator(
		    args.annotations,
		    args.classes,
		    transform_generator=transform_generator,
		    visual_effect_generator=visual_effect_generator,
		    **common_args
		)

		if args.val_annotations:
		    validation_generator = CSVGenerator(
			args.val_annotations,
			args.classes,
			shuffle_groups=False,
			**common_args
		    )
		else:
		    validation_generator = None


def set_defaults(config):
	# Set defaults for backbone.
	if 'backbone' not in config:
		config['backbone'] = {}
	if 'details' not in config['backbone']:
		config['backbone']['details'] = {}

	# Set defaults for generator.
	if 'generator' not in config:
		config['generator'] = {}
	if 'details' not in config['generator']:
		config['generator']['details'] = {}

	# Set the defaults for convert.
	if 'convert' not in config:
		config['convert'] = {}
	if 'nms' not in config['convert']:
		config['convert']['nms'] = True
	if 'class_specific_filter' not in config['convert']:
		config['convert']['class_specific_filter'] = True

	return config


def parse_args(args):
	""" Parse the command line arguments.
	"""
	parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

	parser.add_argument('model_in',                   help='The model to convert.')
	parser.add_argument('model_out',                  help='Path to save the converted model to.')
	parser.add_argument('--config',                   help='Config file.', default=None, type=str)
	parser.add_argument('--backbone',                 help='The backbone of the model to convert.')
	parser.add_argument('--no-nms',                   help='Disables non maximum suppression.',  dest='nms',                   action='store_false')
	parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.', dest='class_specific_filter', action='store_false')
	parser.add_argument('--lite',                     help='Convert to tensorflow lite.',        dest='lite',                  action='store_true')
	parser.add_argument('--savedmodel',               help='Convert to tensorflow SavedModel.',  dest='savedmodel',            action='store_true')

	# Additional config.
	parser.add_argument('--o', help='Additional config, in shape of a dictionary.', type=str, default=None)

	return parser.parse_args(args)


def set_args(config, args):
	# Additional config; start from this so it can be overwirtten by the other command line options.
	if args.o:
		config = parse_additional_options(config, args.o)

	if args.backbone:
		config['backbone']['name'] = args.backbone

	# Convert config.
	config['convert']['nms'] = args.nms
	config['convert']['class_specific_filter'] = args.class_specific_filter


	return config

def csv_list(string):
        return string.split(',')


def main(args=None, config=None):
	# Parse arguments.
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	# Parse the configuration file.
	if config is None:
		config = {}
	if args.config:
		config = parse_yaml(args.config)
	config = set_defaults(config)

	# Apply the command line arguments to config.
	config = set_args(config, args)

	# Set modified tf session to avoid using the GPUs.
	setup_gpu("cpu")

	# Optionally load anchors parameters.
	anchor_params = None
	if 'anchors' in config['generator']['details']:
		anchor_params = parse_anchor_parameters(config['generator']['details']['anchors'])

	# Get the backbone.
	backbone = get_backbone(config)

	# Load the model.
	model = models.load_model(args.model_in, backbone=backbone)

	# Check if this is indeed a training model.
	models.retinanet.check_training_model(model)

	# Convert the model.
	model = models.retinanet.convert_model(
		model,
		config['convert']['nms'],
		class_specific_filter=config['convert']['class_specific_filter'],
		anchor_params=anchor_params
	)

	# Save model.
	if (not args.lite) and (not args.savedmodel):
		model.save(args.model_out)
	elif args.lite:
		print('Converting to tensorflow lite.')
		import tensorflow as tf
		model.layers.pop(0)

		fixed_size_inputs = tf.keras.layers.Input(shape=(1333, 800, 3))
		outputs = model(fixed_size_inputs)
		fixed_size_model = tf.keras.Model(fixed_size_inputs, outputs)

		converter = tf.lite.TFLiteConverter.from_keras_model(fixed_size_model)
		target_spec = tf.lite.TargetSpec(supported_ops = set([tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]))
		converter.target_spec = target_spec
		tflite_model = converter.convert()
	elif args.savedmodel:
		print('Converting to savedmodel.')
		import tensorflow as tf
		tf.saved_model.save(model, args.model_out)




if __name__ == '__main__':
	main()
