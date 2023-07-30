import itertools

import cv2
import numpy as np
import random
import math


def mutual_distance_greater_than(points, epsilon):
    for i in range(3):
        for j in range(i + 1, 3):
            if np.linalg.norm(np.array(points[i]) - np.array(points[j])) <= epsilon:
                return False
    return True


def sample_three_points(D, epsilon):
    while True:
        points = random.sample(list(D), 3)
        if mutual_distance_greater_than(points, epsilon):
            return points


def is_colinear(p1, p2, p3, epsilon=0.001):
    return abs((p2[1] - p1[1]) * (p3[0] - p2[0]) - (p3[1] - p2[1]) * (p2[0] - p1[0])) < epsilon


def calculate_circle_parameters(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    if D == 0:
        return None

    Ux = (((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / D)
    Uy = (((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / D)
    r = np.sqrt((x1 - Ux) ** 2 + (y1 - Uy) ** 2)

    return (Ux, Uy, r)

def is_center_inside_image(a, b, img_shape):
    return 0 <= a < img_shape[1] and 0 <= b < img_shape[0]


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_circle_distance(p, a, b, r):
    return abs(math.sqrt((p[0] - a) ** 2 + (p[1] - b) ** 2) - r)


def is_point_on_circle(p, a, b, r, T2):
    return point_to_circle_distance(p, a, b, r) < T2


def calculate_delta(edge_points, edge_threshold):
    distances = []
    for _ in range(edge_threshold):
        p1, p2 = random.sample(edge_points, 2)
        distances.append(euclidean_distance(p1, p2))
    return np.median(distances)




# Helper functions
def circle_params(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    k12 = (x1 - x2) / (y2 - y1)
    k23 = (x2 - x3) / (y3 - y2)
    j12 = (y1 + y2 - k12 * (x1 + x2)) / 2
    j23 = (y2 + y3 - k23 * (x2 + x3)) / 2

    if k23 == k12 or y2 == y1 or y3 == y2:
        return None

    a = (j12 - j23) / (k23 - k12)
    b = k12 * a + j12
    r = math.sqrt((x2 - a) ** 2 + (y2 - b) ** 2)

    return a, b, r


def distance_to_circle(p, a, b, r):
    px, py = p
    return abs(math.sqrt((px - a) ** 2 + (py - b) ** 2) - r)


def circle_is_candidate(edge_points, a, b, r, delta, lambda_, T1, T2):
    count = 0
    for _ in range(T1):
        p = random.choice(edge_points)
        if distance_to_circle(p, a, b, r) < T2:
            count += 1
            if count > lambda_ * 2 * math.pi * r:
                return True
    return False


def search_true_circle(edge_points, a, b, r, delta, lambda_, epsilon, T3, T2):
    random_sampling_area = [(x, y) for x, y in edge_points if r - T2 <= distance_to_circle((x, y), a, b, r) <= r + T2]
    nc3 = len(random_sampling_area)

    for _ in range(T3):
        p1, p2, p3 = random.sample(random_sampling_area, 3)
        while math.dist(p1, p2) <= epsilon or math.dist(p2, p3) <= epsilon or math.dist(p1, p3) <= epsilon:
            p1, p2, p3 = random.sample(random_sampling_area, 3)

        circle = circle_params(p1, p2, p3)
        if circle is None:
            continue

        a_prime, b_prime, r_prime = circle
        if abs(a - a_prime) > 2 * T2 or abs(b - b_prime) > 2 * T2:
            continue

        evidence_collection_area = [(x, y) for x, y in edge_points if
                                    r - 2 * T2 <= distance_to_circle((x, y), a, b, r) <= r + 2 * T2]
        nc2 = len([p for p in evidence_collection_area if distance_to_circle(p, a_prime, b_prime, r_prime) < T2])

        if nc2 > lambda_ * 2 * math.pi * r:
            return a_prime, b_prime, r_prime

    return None

def is_valid_circle(circle, image_shape, epsilon):
    if circle is None:
        return False

    Ux, Uy, r = circle
    if Ux < 0 or Ux >= image_shape[1] or Uy < 0 or Uy >= image_shape[0]:
        return False

    return True

def calculate_circle_params(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    k12 = (x1 - x2) / (y2 - y1)
    k23 = (x2 - x3) / (y3 - y2)

    j12 = (y1 + y2 - k12 * (x1 + x2)) / 2
    j23 = (y2 + y3 - k23 * (x2 + x3)) / 2

    a = (j12 - j23) / (k23 - k12)
    b = k12 * a + j12
    r = np.sqrt((x2 - a) ** 2 + (y2 - b) ** 2)

    return a, b, r

def point_within_t2(point, circle, T2):
    px, py = point
    a, b, r = circle
    distance = np.abs(np.sqrt((px - a) ** 2 + (py - b) ** 2) - r)
    return distance < T2

def find_true_circle(circle, edge_image, epsilon, T3, M_min):
    a, b, r = circle
    edge_points = np.argwhere(edge_image > 0)
    random_sampling_area = [(x, y) for x, y in edge_points if (r - epsilon) ** 2 <= (x - a) ** 2 + (y - b) ** 2 <= (r + epsilon) ** 2]

    if len(random_sampling_area) < 3:
        return None

    for _ in range(T3):
        random_points = [random_sampling_area[i] for i in np.random.choice(len(random_sampling_area), 3, replace=False)]
        if all(np.linalg.norm(np.array(p1) - np.array(p2)) > epsilon for p1, p2 in itertools.combinations(random_points, 2)):
            a_new, b_new, r_new = calculate_circle_params(*random_points)
            circle_new = (a_new, b_new, r_new)

            evidence_collection_area = [(x, y) for x, y in edge_points if (r - 2 * epsilon) ** 2 <= (x - a) ** 2 + (y - b) ** 2 <= (r + 2 * epsilon) ** 2]
            N_C2 = sum(point_within_t2(point, circle_new, epsilon) for point in evidence_collection_area)

            if N_C2 > M_min:
                return circle_new

    return circle

def edge_detection(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def count_votes_on_circle(edge_points, circle, d):
    a, b, r = circle
    votes = 0
    for point in edge_points:
        x, y = point
        distance = np.sqrt((x - a) ** 2 + (y - b) ** 2)
        if abs(distance - r) <= d:
            votes += 1
    return votes

def calculate_temp3(M_min, N_C4):
    return (M_min / N_C4) ** 3

def calculate_temp4(r):
    return r / 100 if r / 100 > 2 else 2

def auto_canny(image, sigma=0.33):
        #    # Compute the median of image intensities
        v = np.median(image)

        # Apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        #self.save_debug_image('_canny', edged)
        return edged

def fast_accurate_circle_detection(image ,use_binary, sigma, epsilon, k_max, s, d, n_min):

    if use_binary is True:
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank image to draw the filled contours
        edge_image = np.zeros_like(image)

        # Draw the filled contours on the blank image
        cv2.drawContours(edge_image, contours, -1, (255), thickness=1)
    else:
        edge_image = auto_canny(image,sigma)
    edge_points = np.argwhere(edge_image > 0)
    edge_points = [(x[1], x[0]) for x in edge_points]

    k = 0
    detected_circles = []

    while k < k_max and len(edge_points) >= 3:
        # Step 3: Randomly sample three points from edge_points whose mutual distances are greater than epsilon
        p1, p2, p3 = sample_three_points(edge_points, epsilon)

        # Step 4: Calculate the circle parameters determined by these points
        candidate_circle = calculate_circle_parameters(p1, p2, p3)

        # Step 5-8: Check if the center of the circle parameter falls outside the image
        if candidate_circle is None or not is_valid_circle(candidate_circle, image.shape, epsilon):
            k += 1
            continue

        # Step 9-13: Judge whether the circle corresponding to the circle parameter is a candidate circle
        votes = count_votes_on_circle(edge_points, candidate_circle, d)
        if votes <= n_min:
            k += 1
            continue

        # Step 14-18: Search for a true circle based on the information of the candidate circle
        true_circle = find_true_circle(candidate_circle, edge_image, epsilon, d, n_min)

        if true_circle is None:
            k += 1
            continue

        # Step 19: Improve the accuracy of the true circle and determine the final true circle
        # skip

        # Step 20-26: Judge if the number of detected circles is less than the preset number
        if len(detected_circles) < s:
            detected_circles.append(true_circle)

            # Step 22: Delete the points on the final true circle from the array D
            a, b, r = true_circle
            edge_points = [point for point in edge_points if np.sqrt((point[0] - a) ** 2 + (point[1] - b) ** 2) > r + epsilon]

            # Step 23: Reset k=0
            k = 0
        else:
            # Step 25: Terminate the detection
            break

    # Step 28: Terminate the detection
    return detected_circles


