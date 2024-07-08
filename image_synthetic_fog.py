import cv2, math
import sys


def atomization(file_img, A, beta):
    img_f = file_img / 255.0  # Normalization
    (row, col, chs) = file_img.shape

    A = float(A)
    beta = float(beta)

    size = math.sqrt(max(row, col))  # Atomization Size
    center = (row // 2, col // 2)  # The center of the fog
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f * 255


def main(file_img, A, beta, save_img):
    file_img = cv2.imread(file_img)
    foggy_result = atomization(file_img, A, beta)
    cv2.imwrite(save_img, foggy_result, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == "__main__":
    file_img = sys.argv[1]  # Path to the image
    A = sys.argv[2]  # Brightness of the Fog
    beta = sys.argv[3]  # Thickness of the fog
    save_img = sys.argv[4]  # Save path of the image
    main(file_img, A, beta, save_img)

