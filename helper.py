import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num



def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))



def gen_batch_function(data_folder, image_shape, num_classes):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

        background_color = np.array([255, 0, 0])
        road_color = np.array([255, 0, 255])
        other_road_color = np.array([0, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):

            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:

                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                # separating ground truth channels
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_rd = np.all(gt_image == road_color, axis=2)
                gt_o_rd = np.all(gt_image == other_road_color, axis=2)

                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_rd = gt_rd.reshape(*gt_rd.shape, 1)
                gt_o_rd = gt_o_rd.reshape(*gt_o_rd.shape, 1)

                if num_classes == 3:
                    gt_image = np.concatenate((gt_bg, gt_rd, gt_o_rd), axis=2)

                else:
                    # unfortunately, this doesn't work (but why?...)
                    # gt_rd = gt_rd + gt_o_rd
                    # gt_image = np.concatenate((gt_bg, gt_rd), axis=2)

                    # so using the original code
                    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn



def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape, params={}):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    (height, width) = image_shape

    # extracting relevant parameters
    debug = params['debug'] if 'debug' in params else False
    debug_dir = params['debug_dir'] if 'debug_dir' in params else './debug'
    num_classes = params['num_classes'] if 'num_classes' in params else 2
    probability_threshold = params['probability_threshold'] if 'probability_threshold' in params else 0.5
    test_file_pattern = params['test_file_pattern'] if 'test_file_pattern' in params else '*.png'

    # cleaning up
    if debug:
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)

    # processing files
    for image_file in glob(os.path.join(data_folder, 'image_2', test_file_pattern)):

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})

        # masking the "basic" road pixels (pixels which have been marked explicitly as "road")
        im_softmax_rd = im_softmax[0][:, 1].reshape(height, width)
        segmentation_rd = (im_softmax_rd > probability_threshold).reshape(height, width, 1)

        # creating the mask based on road pixels
        mask = np.dot(segmentation_rd, np.array([[0, 255, 0, 127]]))

        if debug:
            np.save(os.path.join(debug_dir, 'im_softmax'), im_softmax)
            np.save(os.path.join(debug_dir, 'im_softmax_rd'), im_softmax_rd)
            np.save(os.path.join(debug_dir, 'segmentation_rd'), segmentation_rd)
            np.save(os.path.join(debug_dir, 'mask_rd'), mask)

        # extrawurst for three classes
        if num_classes == 3:

            # basic case: explicit "other road" pixel values
            im_softmax_o_rd = im_softmax[0][:, 2].reshape(height, width)
            segmentation_o_rd = (im_softmax_o_rd > probability_threshold).reshape(height, width, 1)

            # an "other road" is just a road as well, so we might be confused in certain cases, where
            # the _individual_ probabilities of "road" and "other road" lie below the threshold, but the
            # _sum_ of these probabilities might sill be over the threshold -- meaning that it's a road pixel

            # creating a "union" road segmentation
            im_softmax_rd_union = im_softmax_rd + im_softmax_o_rd
            segmentation_rd_union = (im_softmax_rd_union > probability_threshold).reshape(height, width, 1)

            # we have to decide which road pixel belongs to which class
            segmentation_rd_dominant = (im_softmax_rd >= im_softmax_o_rd).reshape(height, width, 1)
            segmentation_o_rd_dominant = (im_softmax_o_rd > im_softmax_rd).reshape(height, width, 1)

            segmentation_rd_dominant_union = (segmentation_rd_union) & (segmentation_rd_dominant)
            segmentation_o_rd_dominant_union = (segmentation_rd_union) & (segmentation_o_rd_dominant)

            # updating the mask. using different colors for the different cases (mainly for debug purposes)
            mask_o_rd = np.dot(segmentation_o_rd, np.array([[0, 0, 255, 127]]))
            mask_rd_dominant = np.dot(segmentation_rd_dominant_union, np.array([[0, 255, 127, 127]]))
            mask_o_rd_dominant = np.dot(segmentation_o_rd_dominant_union, np.array([[0, 127, 255, 127]]))

            # yeah, I know, we'll definitely have values over 255 (...)
            mask += mask_o_rd
            mask += mask_rd_dominant
            mask += mask_o_rd_dominant

            if debug:
                np.save(os.path.join(debug_dir, 'im_softmax_o_rd'), im_softmax_o_rd)
                np.save(os.path.join(debug_dir, 'segmentation_o_rd'), segmentation_o_rd)
                np.save(os.path.join(debug_dir, 'im_softmax_rd_union'), im_softmax_rd_union)
                np.save(os.path.join(debug_dir, 'segmentation_rd_union'), segmentation_rd_union)

                np.save(os.path.join(debug_dir, 'segmentation_rd_dominant'), segmentation_rd_dominant)
                np.save(os.path.join(debug_dir, 'segmentation_o_rd_dominant'), segmentation_o_rd_dominant)
                np.save(os.path.join(debug_dir, 'segmentation_rd_dominant_union'), segmentation_rd_dominant_union)
                np.save(os.path.join(debug_dir, 'segmentation_o_rd_dominant_union'), segmentation_o_rd_dominant_union)

                np.save(os.path.join(debug_dir, 'mask_o_rd'), mask_o_rd)
                np.save(os.path.join(debug_dir, 'mask_rd_dominant'), mask_rd_dominant)
                np.save(os.path.join(debug_dir, 'mask_o_rd_dominant'), mask_o_rd_dominant)
                np.save(os.path.join(debug_dir, 'mask_final'), mask)

        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(sess, logits, keep_prob, input_image, params={}):

    # extracting relevant parameters
    runs_dir = params['runs_dir'] if 'runs_dir' in params else './runs'
    data_dir = params['data_dir'] if 'data_dir' in params else './data'
    image_shape = params['image_shape'] if 'image_shape' in params else (160, 576)

    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape, params)

    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    return output_dir
