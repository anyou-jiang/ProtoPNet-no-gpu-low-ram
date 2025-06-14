import cv2
import os

bounding_boxes_filepath = './datasets/CUB_200_2011/bounding_boxes.txt'
images_filepath = './datasets/CUB_200_2011/images.txt'
train_test_split_filepath = './datasets/CUB_200_2011/train_test_split.txt'

bounding_boxes_file = open(bounding_boxes_filepath)
images_file = open(images_filepath)
train_test_split_file = open(train_test_split_filepath)

while True:
    image_line = images_file.readline()
    bounding_boxes_line = bounding_boxes_file.readline()
    train_test_line = train_test_split_file.readline()

    if (not image_line) or (not bounding_boxes_line) or (not train_test_line):
        if any([image_line, bounding_boxes_line, train_test_line]):
            raise RuntimeError('the number of lines in images.txt and in bounding_boxes.txt should be the same.')
        break

    image_line_parts = image_line.split()
    bounding_boxes_line_parts = bounding_boxes_line.split()
    train_test_line_parts = train_test_line.split()

    if int(image_line_parts[0]) != int(bounding_boxes_line_parts[0]) \
            or int(image_line_parts[0]) != int(train_test_line_parts[0]):
        raise RuntimeError('the image id is not aligned!')

    image_filepath = image_line_parts[1]
    bounding_x, bounding_y, bounding_w, bounding_h = (float(val) for val in bounding_boxes_line_parts[1:5])
    is_training_image = bool(int(train_test_line_parts[1]))

    image_fullpath = './datasets/CUB_200_2011/images/{}'.format(image_filepath)
    image = cv2.imread(image_fullpath)

    x = int(bounding_x)
    y = int(bounding_y)
    w = int(bounding_w)
    h = int(bounding_h)

    crop_image = image[y:y + h, x:x + w]

    if is_training_image:
        cropped_image_fullpath = './datasets/cub200_cropped/train_cropped/{}'.format(image_filepath)
    else:
        cropped_image_fullpath = './datasets/cub200_cropped/test_cropped/{}'.format(image_filepath)

    head_tail = os.path.split(cropped_image_fullpath)
    if not os.path.exists(head_tail[0]):
        os.makedirs(head_tail[0])
    if not cv2.imwrite(cropped_image_fullpath, crop_image):
        raise RuntimeError('failed in generating cropped images.')

