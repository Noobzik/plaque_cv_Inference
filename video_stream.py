from __future__ import division
import time
import pytesseract
import tensorflow as tf
import numpy as np
import cv2
import colorsys
import socket
import struct
import math
import boto3

# Infos de connexion Kinesis
AWS_KINESIS_USER_KEY = ""
AWS_KINESIS_PASS_KEY = ""
AWS_KINESIS_STREAM_NAME = ""
# Adresse IP du site pour renvoyer le feed UDP
AWS_EC2_ADDRESS_IP_SITE = ""
# Infos de connexion S3
AWS_S3_USER_KEY = ""
AWS_S3_USER_PASS = ""
AWS_S3_BUCKET_NAME = ""

class FrameSegment(object):
    """ 
    Object to break down image frame segment
    if the size of image exceed maximum datagram size 
    """
    MAX_DGRAM = 2 ** 16
    MAX_IMAGE_DGRAM = MAX_DGRAM - 64  # extract 64 bytes in case UDP frame overflown

    def __init__(self, sock, port, addr="127.0.0.1"):
        self.s = sock
        self.port = port
        self.addr = addr

    def udp_frame(self, img):
        """ 
        Compress image and Break down
        into data segments 
        """
        compress_img = cv2.imencode('.jpg', img)[1]
        dat = compress_img.tobytes()
        size = len(dat)
        count = math.ceil(size / self.MAX_IMAGE_DGRAM)
        array_pos_start = 0
        while count:
            array_pos_end = min(size, array_pos_start + self.MAX_IMAGE_DGRAM)
            self.s.sendto(struct.pack("B", count) +
                          dat[array_pos_start:array_pos_end],
                          (self.addr, self.port)
                          )
            array_pos_start = array_pos_end
            count -= 1


# Complexité : (n*m)*nombre de frames*boxes
def color_detection(min_color, max_color, hsv_img, img):
    low_color = np.array(min_color, np.uint8)
    high_color = np.array(max_color, np.uint8)
    color_mask = cv2.inRange(hsv_img, low_color, high_color)

    height, width, channels = img.shape
    count_color = 0

    for h in range(height):
        for w in range(width):
            if color_mask[h][w] == 255:
                count_color += 1
    return count_color


def draw_bbox(image, bboxes, classes, allowed_classes, show_label=True, show_color=False):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    list_name_color = ["red", "green", "blue", "white", "black", "gray", "yellow", "orange"]
    list_count_color = [0, 0, 0, 0, 0, 0, 0, 0]
    count_min_color = [[94, 80, 2], [35, 52, 72], [170, 70, 50], [0, 0, 175], [0, 0, 0], [0, 10, 70], [20, 100, 100],
                       [5, 150, 150]]
    count_max_color = [[180, 255, 255], [102, 255, 255], [126, 255, 255], [172, 111, 255], [180, 255, 60],
                       [179, 50, 255], [40, 255, 255], [15, 235, 250]]

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    crop_img = None
    name_color = ""

    class_ind = -1
    for i in range(num_boxes[0]):
        text = ""

        list_count_color = [0, 0, 0, 0, 0, 0, 0, 0]
        if int(out_classes[0][i]) < 0 or (int(out_classes[0][i]) > num_classes):
            continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        class_name = classes[class_ind]
        # print(score)
        # check if class is in allowed classes
        crop_img = image[int(coor[0]):int(coor[2]), int(coor[1]):int(coor[3])]

        if class_name not in allowed_classes:
            continue
        else:
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1 = (int(coor[1]), int(coor[0]))
            c2 = (int(coor[3]), int(coor[2]))
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                if show_color:
                    hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                    list_count_color = [0, 0, 0, 0, 0, 0, 0, 0]
                    for c in range(len(list_count_color)):
                        list_count_color[c] = color_detection(count_min_color[c], count_max_color[c], hsv_img, crop_img)
                    max_value = max(list_count_color)
                    max_index = list_count_color.index(max_value)
                    name_color = list_name_color[max_index]
                    bbox_mess = '%s: %.2f %s' % (classes[class_ind], score, name_color)
                else:

                    text = pytesseract.image_to_string(crop_img,
                                                       config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- --psm 7 --oem 3')
                    print("OCR : " + text)
                    bbox_mess = '%s: %.2f %s' % (classes[class_ind], score, text)

                t_size = cv2.getTextSize(bbox_mess + "null", 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (int(c3[0]), int(c3[1])), bbox_color, -1)  # filled

                cv2.putText(image, bbox_mess, (c1[0], int(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

                print("------------------")
                print("List count color", list_count_color)
                print("Max count : ", max(list_count_color))
                if show_color:
                    print("Name color : ", name_color)
    if show_color:
        return image, crop_img, classes[class_ind], name_color, None
    elif class_ind != -1 and show_color is False:
        return image, crop_img, classes[class_ind], name_color, text
    else:
        return image, crop_img, None, None, None


def dump_buffer(s):
    """ Emptying buffer frame """
    MAX_DGRAM = 2 ** 16
    while True:
        seg, addr = s.recvfrom(MAX_DGRAM)
        print(seg[0])
        if struct.unpack("B", seg[0:1])[0] == 1:
            print("finish emptying buffer")
            break


if __name__ == '__main__':

    kvs = boto3.client("kinesisvideo",
                       aws_access_key_id=AWS_KINESIS_USER_KEY,
                       aws_secret_access_key=AWS_KINESIS_PASS_KEY,
                       region_name='us-east-1')
    # Grab the endpoint from GetDataEndpoint
    endpoint = kvs.get_data_endpoint(
        APIName="GET_HLS_STREAMING_SESSION_URL",
        StreamName=AWS_KINESIS_STREAM_NAME
    )['DataEndpoint']

    print(endpoint)

    # # Grab the HLS Stream URL from the endpoint
    kvam = boto3.client("kinesis-video-archived-media",
                        endpoint_url=endpoint,
                        aws_access_key_id=AWS_KINESIS_USER_KEY,
                        aws_secret_access_key=AWS_KINESIS_PASS_KEY,
                        region_name='us-east-1')
    url = kvam.get_hls_streaming_session_url(
        StreamName=AWS_KINESIS_STREAM_NAME,
        PlaybackMode="LIVE",
    )['HLSStreamingSessionURL']

    # Chargement du modele et configuration de la RTX

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    model_1 = tf.keras.models.load_model("./yolo-car-416", compile=False)
    model_2 = tf.keras.models.load_model("./yolo-lpr", compile=False)

    # Video local

    # cap = cv2.VideoCapture('031805_1415_1430.dat')
    # cap = cv2.VideoCapture('cars.avi')

    # Webcam direct feed
    # cap = cv2.VideoCapture(0)

    # Kinesis feed
    cap = cv2.VideoCapture(url)
    print("Before Socket Stream")
    # Socket Stream
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 55055
    fs = FrameSegment(s, port, addr=AWS_EC2_ADDRESS_IP_SITE)
    print("After socket stream setup")

    #    cap.set(3, 416)
    #    cap.set(4, 416)
    class_car = ["Car", "Truck"]
    class_lpr = ["Lpr"]

    frame_rate = 3
    prev = 0

    # Live Feed

    while True:
        time_elapsed = time.time() - prev
        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow("Video", frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time_elapsed > 1. / frame_rate:
            prev = time.time()
            image_data = cv2.resize(frame, (416, 416))
            image_data = image_data / 255.

            images_data = []
            for i in range(1):
                images_data.append(image_data)
            images_data = np.asarray(images_data).astype(np.float32)
            batch_data = tf.constant(images_data)
            pred_bbox = model_1.predict_on_batch(batch_data)

            boxes = pred_bbox[:, :, 0:4]
            pred_conf = pred_bbox[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.1,
                score_threshold=0.75
            )

            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

            # En test -> Car + Truck

            result, cropped, car_type, car_color, _ = draw_bbox(frame, pred_bbox, show_color=True, classes=class_car,
                                                                allowed_classes=class_car)

            if cropped is not None:
                images_data_2 = []

                image_lpr = cv2.resize(frame, (416, 416))
                image_lpr = image_lpr / 255.

                for i in range(1):
                    images_data_2.append(image_lpr)
                images_data_2 = np.asarray(images_data_2).astype(np.float32)
                batch_data_2 = tf.constant(images_data_2)

                # >Run prediction

                pred_bbox_2 = model_2.predict_on_batch(batch_data_2)

                boxes_2 = pred_bbox_2[:, :, 0:4]
                pred_conf_2 = pred_bbox_2[:, :, 4:]

                boxes_2, scores_2, classes_2, valid_detections_2 = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes_2, (tf.shape(boxes_2)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf_2, (tf.shape(pred_conf_2)[0], -1, tf.shape(pred_conf_2)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=0.1,
                    score_threshold=0.75
                )
                pred_bbox_2 = [boxes_2.numpy(), scores_2.numpy(), classes_2.numpy(), valid_detections_2.numpy()]
                result, cropped_3, _, _, car_plaque = draw_bbox(frame, pred_bbox_2, show_color=False, classes=class_lpr,
                                                                allowed_classes=class_lpr)
                text = ""

            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#             cv2.imshow("result", result)

            # Write live feed as UDP SOCKET
            res = time.localtime()
            # Write output
            if car_type:
                try :
                    car_plaque
                except NameError: car_plaque = ""

                cv2.imwrite('outputfeed/{}-{}-{}--{}:{}:{}--{}-{}-{}.jpg'.format(
                    res.tm_year,
                    res.tm_mon,
                    res.tm_mday,
                    res.tm_hour,
                    res.tm_min,
                    res.tm_sec,
                    car_type,
                    car_color,
                    car_plaque), result)
#             cv2.imshow('image', result)
            fs.udp_frame(result)

    cap.release()
    cv2.destroyAllWindows()
