import cv2
import numpy as np
import sys


def get_noise(file_img, value):
    noise = np.random.uniform(0, 256, file_img.shape[0:2])
    # Control the noise level by taking a floating point number and keeping only the largest portion as noise
    v = int(value) * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # Noise does initial blurring.
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    return noise


def rain_blur(noise, length, angle, width):
    length = int(length)
    angle = int(angle)
    width = int(width)

    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # Generate focus matrix
    k = cv2.warpAffine(dig, trans, (length, length))  # Generation of fuzzy kernels
    k = cv2.GaussianBlur(k, (width, width), 0)

    blurred = cv2.filter2D(noise, -1, k)  # Using the rotated kernel just obtained, the filtering

    # Convert to the 0-255 interval
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


def alpha_rain(rain, file_img, beta=0.8):
    rain = np.expand_dims(rain, 2)

    rain_result = file_img.copy()  # Copy a mask
    rain = np.array(rain, dtype=np.float32)  # The data type is changed to a floating-point number
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]


    return rain_result


def main(file_img, angle, value, width, length, save_img):
    file_img = cv2.imread(file_img)
    noise = get_noise(file_img, value)
    rain = rain_blur(noise, length, angle, width)
    rain_result = alpha_rain(rain, file_img, beta=0.6)
    cv2.imwrite(save_img, rain_result, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == "__main__":
    file_img = sys.argv[1]  # Path to the image
    angle = sys.argv[2]  # Angle of raindrops
    value = sys.argv[3]  # Number of raindrops
    width = sys.argv[4]  # width of raindrops
    length = sys.argv[5]  # length of raindrops
    save_img = sys.argv[6]  # Save path of the image
    main(file_img, angle, value, width, length, save_img)
