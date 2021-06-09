import cv2
import numpy as np

def block_matching(input_1, input_2, kernel_size_1, kernel_size_2, threshold_var):
    assert input_1.shape == input_2.shape
    n1 = (kernel_size_1 - 1) // 2
    n2 = (kernel_size_2 - 1) // 2
    height, width = input_1.shape[0:2]
    map = np.zeros((height, width, 3), dtype=int)
    for y1 in range(n1+n2, height-n1-n2):
        for x1 in range(n1+n2, width-n1-n2):
            block_1 = input_1[y1-n1:y1+n1+1, x1-n1:x1+n1+1]
            variance = np.var(block_1)
            if variance > threshold_var:
                min_sum = kernel_size_1*kernel_size_1*255
                for y2 in range(y1-n2, y1+n2+1):
                    for x2 in range(x1-n2, x1+n2+1):
                        block_2 = input_2[y2-n1:y2+n1+1, x2-n1:x2+n1+1]
                        diff_sum = np.sum(np.abs(block_1 - block_2))
                        if(diff_sum < min_sum):
                            min_sum = diff_sum
                            map[y1][x1][0] = diff_sum
                            map[y1][x1][1] = y2
                            map[y1][x1][2] = x2

    return map

def image_loader(input_rgb, scale):
    # input_rgb = cv2.imread(img_path)
    input_gray = cv2.cvtColor(input_rgb, cv2.COLOR_BGR2GRAY)
    height, width = input_gray.shape[0:2]
    input_gray = cv2.resize(input_gray, (width//scale , height//scale))
    return input_rgb, input_gray

def optical_flow(img_1, img_2):
    scale = 8
    kernel_1 = 5
    kernel_2 = 15
    threshold = 255*kernel_1*kernel_1*0.1
    threshold_var = 2000
    input_rgb_1, input_gray_1 = image_loader(img_1, scale)
    input_rgb_2, input_gray_2 = image_loader(img_2, scale)
    print("input size:", input_gray_1.shape[0:2])
    map = block_matching(input_gray_1, input_gray_2, kernel_1, kernel_2, threshold_var)
    output_rgb = cv2.resize(input_rgb_1, (input_rgb_1.shape[1]//scale, input_rgb_1.shape[0]//scale))

    for y_1 in range(map.shape[0]):
        for x_1 in range(map.shape[1]):
            val = map[y_1][x_1][0]
            y_2 = map[y_1][x_1][1]
            x_2 = map[y_1][x_1][2]
            if(val <= threshold and x_2 != 0):
                xy_1 = (x_1, y_1)
                xy_2 = (x_2, y_2)
                cv2.line(output_rgb, xy_1, xy_2, (255,0,0), 1)

    return output_rgb

if __name__ == "__main__":
    img_1 = cv2.imread('data/IMG_9742.jpeg')
    img_2 = cv2.imread('data/IMG_9743.jpeg')
    output_rgb = optical_flow(img_1, img_2)
    cv2.imshow("output_rgb", output_rgb)
    cv2.waitKey(0)


