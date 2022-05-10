import os
import glob
import numpy as np
import cv2


def main():
    # get image and labels filenames
    file_path = os.path.join("Dataset", "invoice",
                             "20220307_yolo_trainforadjust")
    image_save_path = os.path.join("20220307_Invoice_Normalization", "output")
    img_filenames = glob.glob(os.path.join(file_path, "*.jpg"))

    scale_coefficient = 1500
    angle_coefficient = 3.525
    trim_dx1 = scale_coefficient*0.02
    trim_dy1 = scale_coefficient*0.03
    trim_dx2 = int(scale_coefficient*0.7)
    trim_dy2 = int(scale_coefficient*1.2)

    for img_filename in img_filenames:
        # label and image save filename
        filename = os.path.splitext(os.path.basename(img_filename))[0]
        label_filename = os.path.join(file_path, "labels", filename) + ".txt"

        # read image and label info
        img = cv2.imdecode(np.fromfile(img_filename, dtype=np.uint8), -1)
        label = np.loadtxt(label_filename)

        # normalization
        img_shape = img.shape
        x1 = int(img_shape[1]*label[0][1])
        y1 = int(img_shape[0]*label[0][2])
        x2 = int(img_shape[1]*label[1][1])
        y2 = int(img_shape[0]*label[1][2])
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        dx = x1 - x2
        dy = y1 - y2

        angle = np.arctan(dy/dx)*180/np.pi - angle_coefficient
        length = (dx**2 + dy**2)**0.5
        scale_ratio = scale_coefficient / length
        size = (int(img.shape[1]*scale_ratio),
                int(img.shape[0]*scale_ratio))

        M = cv2.getRotationMatrix2D((y1, x1), angle, 1)
        img = cv2.warpAffine(
            img, M, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        trim_x1 = int(y1*scale_ratio - trim_dx1)
        trim_x2 = trim_x1 + trim_dx2
        trim_y1 = int(x1*scale_ratio - trim_dy1)
        trim_y2 = trim_y1 + trim_dy2
        img = img[trim_x1:trim_x2, trim_y1:trim_y2]

        # export result
        img_save_filename = os.path.join(
            image_save_path, filename) + "_normalized.jpg"
        cv2.imencode(".jpg", img)[1].tofile(img_save_filename)
        print('COMPLETE :', filename)


if __name__ == "__main__":
    main()
