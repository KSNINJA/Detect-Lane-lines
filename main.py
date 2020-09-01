import cv2
import numpy as np

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150)

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250), ]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(mask, img)
    return masked_img

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def average_slope_intercept(img, lines):
    left_lines =[]
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        (slope, intercept) = np.polyfit((x1, x2), (y1, y2), 1)
        if slope>0:
            right_lines.append((slope, intercept))
        else:
            left_lines.append((slope, intercept))
    avg_left_line_params = np.average(left_lines, 0)
    avg_right_line_params = np.average(right_lines, 0)
    left_line_coord = get_coordinates(img, avg_left_line_params)
    right_line_coord = get_coordinates(img, avg_right_line_params)
    return np.array([left_line_coord, right_line_coord])

def get_coordinates(img, line_params):
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - line_params[1])/line_params[0])
    x2 = int((y2 - line_params[1]) / line_params[0])
    return (x1, y1), (x2, y2)

cap = cv2.VideoCapture("input.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
result = cv2.VideoWriter('output.mp4', fourcc, 10.0, size)
while cap.isOpened():
    _, frame = cap.read()
    canny_img = canny(frame)
    masked_img = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(masked_img, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=5)
    average_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(masked_img, average_lines)
    resized_image = np.zeros_like(frame)  #
    resized_image[:, :, 0] = line_image
    required_img = cv2.addWeighted(frame, 0.8, resized_image, 1, 1)
    cv2.imshow("result", required_img)
    result.write(required_img)
    cv2.waitKey(10)
