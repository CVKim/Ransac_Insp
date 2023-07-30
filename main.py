import cv2
from pathlib import Path
import numpy as np
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit
from fast_circle_detection import fast_accurate_circle_detection


def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        # Generate color with equally spaced hue values in HSV color space
        hue = int(255 * i / num_colors)
        saturation = 255
        value = 255
        hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)

        # Convert HSV color to BGR color
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr_color)))

    return colors


def create_clustered_image(outside_points, labels, circle_mask_shape):
    # Create a blank canvas with the same size as the input mask
    result_image = np.zeros((*circle_mask_shape, 3), dtype=np.uint8)

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    # Generate colors based on the number of unique labels
    num_colors = len(unique_labels)
    # Define colors for different clusters
    colors = generate_colors(num_colors)

    # Map labels to colors
    label_to_color = dict(zip(unique_labels, colors))

    # Draw clustered points on the canvas
    for point, label in zip(outside_points, labels):
        if label != -1:  # Ignore noise points with label -1
            x, y = point
            color = label_to_color[label]
            result_image[y, x] = color

    # Save the result image
    return result_image


def fit_curve_and_calculate_curvature(cluster_points):
    # Fit the curve using the quadratic function
    x = cluster_points[:, 0]
    y = cluster_points[:, 1]

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    params, _ = curve_fit(quadratic, x, y)

    # Calculate the curvature for each point
    a, b, c = params
    curvatures = []
    for x_value in x:
        f_prime = 2 * a * x_value + b
        f_double_prime = 2 * a
        curvature = abs(f_double_prime) / (1 + f_prime ** 2) ** (3 / 2)
        curvatures.append(curvature)

    # Calculate the average curvature for the cluster
    avg_curvature = np.mean(curvatures)

    return avg_curvature


def cluster_outside_points(outside_points, eps=10, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(outside_points)
    return clustering.labels_


def compute_circle_curvature(circle_radius):
    return 1 / circle_radius


def is_defective(avg_curvature, circle_curvature_in, tolerance=0.001):
    return abs(avg_curvature - circle_curvature_in) / abs(circle_curvature_in) > tolerance


class ContactLensDetector:
    def __init__(self, original_path_in, need_to_save_img=True):
        self.original_path = original_path_in
        self.need_to_save_img = need_to_save_img

    def draw_circle(self, image, circle, color=(0, 0, 255), thick=1):
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.circle(image_color, (int(circle[0]), int(circle[1])), int(circle[2]), color, thickness=thick)
        self.save_debug_image('circle_img', image_color)

    def find_circle(self, image):
        # if input image is grayscale image, sigma is used for canny edge detection.
        sigma = 0.8
        epsilon = 300
        k_max = 50
        s = 10
        d = 1
        n_min = 300
        #
        # Call the fast_accurate_circle_detection function
        detected_circles = fast_accurate_circle_detection(image, True, sigma, epsilon, k_max, s, d, n_min)

        lens_mask = []

        if detected_circles is None:
            print("circle is not found on the image : {}".format(original_path))

        for circle in detected_circles:
            lens_mask = lens_detector.create_circle_mask(eqhist2, circle)
            lens_detector.draw_circle(lens_img_rough, circle, color=(0, 0, 255), thick=1)

        return lens_mask, circle

    def create_circle_mask(self, image, circle_in, increment=0, postfix='circle_mask'):
        # Create a mask for the lens
        mask = np.zeros_like(image)
        center = (int(circle_in[0]), int(circle_in[1]))
        radius = circle_in[2] + increment
        cv2.circle(mask, center, int(radius), 255, -1)
        self.save_debug_image(postfix, mask)

        return mask

    def save_image(self, output_dir, file_name, img):
        output_path = Path(output_dir) / file_name
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), img)

    def save_debug_image(self, postfix, img):
        if self.need_to_save_img is False:
            return

        output_dir = './debug'
        debug_name = self.original_path.stem + '_' + postfix + '.bmp'

        self.save_image(output_dir, debug_name, img)

    def save_result_image(self, postfix, img):

        output_dir = './result'
        debug_name = self.original_path.stem + '_' + postfix + '.bmp'

        self.save_image(output_dir, debug_name, img)

    def remove_gradation(self, img_in, kernel_size=15):
        # Apply Gaussian blur with a larger kernel to remove smooth gradients
        large_blur = cv2.GaussianBlur(img_in, (kernel_size, kernel_size), 0)
        grad_removed = cv2.subtract(img_in, large_blur)
        # grad_removed = cv2.GaussianBlur(grad_removed, (5, 5), 0)
        self.save_debug_image('gradation_removed', grad_removed)

        return grad_removed

    # Other methods go here

    def remove_noise_median(self, image, kernel_size=5):
        median_filtered = cv2.medianBlur(image, kernel_size)
        self.save_debug_image('median_filtered', median_filtered)
        return median_filtered

    def remove_noise_bilateral(self, image, kernel_size=5):
        bilateral = cv2.bilateralFilter(image, kernel_size, 75, 75)
        self.save_debug_image('bl_filtered', bilateral)
        return bilateral

    def binarize(self, image, size):
        img_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, size, 1)
        median_filtered = cv2.medianBlur(img_thresh, 3)
        self.save_debug_image('binarized', median_filtered)
        return median_filtered

    def morph_close(self, image, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        close_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        self.save_debug_image('closed', close_img)
        return close_img

    def morph_open(self, image, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        self.save_debug_image('open', result)
        return result

    def auto_canny(self, image, sigma_in=0.33):
        #    # Compute the median of image intensities
        v = np.median(image)

        # Apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma_in) * v))
        upper = int(min(255, (1.0 + sigma_in) * v))
        edged = cv2.Canny(image, lower, upper)
        self.save_debug_image('canny', edged)
        return edged

    def edge_detection(self, image, sigma=0.1):
        edges = self.auto_canny(image, sigma)
        self.save_debug_image('edges', edges)
        return edges

    def dog_filter(self, image, sigma1=3, sigma2=5):
        dog = cv2.GaussianBlur(image, (0, 0), sigma2) - cv2.GaussianBlur(image, (0, 0), sigma1)
        self.save_debug_image('dog', dog)

    def otsu_threshold(self, image, postfix='otsu'):
        # Otsu's thresholding
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.save_debug_image(postfix, thresh)

        return thresh

    def filter_image_with_mask(self, image, mask_in):
        masked_image = cv2.bitwise_and(image, image, mask=mask_in)
        invt_mask = cv2.bitwise_not(mask_in)
        masked_image = cv2.bitwise_or(invt_mask, masked_image)
        self.save_debug_image('filter_image_with_mask', masked_image)
        return masked_image

    def unwrap_circle(self, circle, image, enlarge_pixel):
        center = (circle[0], circle[1])
        radius = circle[2] + enlarge_pixel
        x, y, r = int(center[0]), int(center[1]), int(radius)
        cropped_image = image[y - r:y + r, x - r:x + r]
        self.save_debug_image('cropped_image', cropped_image)

    def eqhist(self, image, postfix='eqhist'):
        dst = cv2.equalizeHist(image)
        self.save_debug_image(postfix, dst)
        return dst

    def fill_gaps_with_contours(self, edge_image, min_points=10):

        # 컨투어 추출
        contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank image to draw the filled contours
        mask = np.zeros_like(edge_image)

        # Draw the filled contours on the blank image
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=2)

        self.save_debug_image('fill_gap', mask)

        return mask

    def extract_closed_loop(self, image):
        # Apply connected component labeling
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

        # Find the largest connected component (excluding the background)
        largest_component_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Create a mask with the largest connected component
        closed_loop_mask = np.zeros(image.shape, dtype=np.uint8)
        closed_loop_mask[labels == largest_component_label] = 255

        # Create a new image containing only the largest connected component
        closed_loop_image = cv2.bitwise_and(image, closed_loop_mask)
        self.save_debug_image('closed_loop_image', closed_loop_image)
        return closed_loop_image

    def find_thick_ring(self, image, circle_mask):

        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        min_area = 100
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

        contour_mask = np.zeros(circle_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

        intersection = np.logical_and(circle_mask, contour_mask)
        union = np.logical_or(circle_mask, contour_mask)
        iou = np.sum(intersection) / np.sum(union)
        self.save_debug_image('contour_mask', contour_mask)

        closed_loop_image = []
        found_thick_ring = False
        if iou < 0.9:
            print("IOU value is below threshold: {}".format(iou))
            print("ring thickness is not enough")
            found_thick_ring = False
        else:
            print("IOU value is large than threshold: {}".format(iou))
            print("ring thickness is enough")
            found_thick_ring = True

        return found_thick_ring

    def erode_mask(self, image, kernel_size):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded_mask = cv2.erode(image, kernel, iterations=1)
        self.save_debug_image('erode_mask', eroded_mask)
        return eroded_mask

    def unsharp_mask(self, image, kernel_size):
        # Apply Gaussian blur to the image
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # Subtract the blurred image from the original image to create the unsharp mask
        unsharp_mask = cv2.subtract(image, blurred)

        # Add the unsharp mask back to the original image to sharpen it
        sharpened = cv2.addWeighted(image, 1.5, unsharp_mask, -0.5, 0)

        self.save_debug_image('sharpen_usm', sharpened)
        return sharpened

    def sharpen_lap(self, image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        # Convert the Laplacian image back to uint8 format
        laplacian = np.uint8(np.absolute(laplacian))
        cv2.normalize(laplacian, laplacian, 0, 255, cv2.NORM_MINMAX)

        # Add the Laplacian image back to the original image to sharpen it
        sharpened = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)

        self.save_debug_image('sharpen_lap', sharpened)
        return sharpened

    def fill_from_seed(self, binary_img):
        nonzero = cv2.findNonZero(binary_img)

        if nonzero is None:
            seed_point = (binary_img.shape[1] // 2, binary_img.shape[0] // 2)
        else:
            cx = np.mean(nonzero[:, 0, 0])
            cy = np.mean(nonzero[:, 0, 1])
            seed_point = (int(cx), int(cy))

        # floodfill 수행
        mask = np.zeros((binary_img.shape[0] + 2, binary_img.shape[1] + 2), np.uint8)
        cv2.floodFill(binary_img, mask, seed_point, 255)
        binary_img = self.morph_open(binary_img, 5)
        self.save_debug_image('flood', binary_img)
        # 결과 반환
        return binary_img, seed_point

    def upscale_img(self, image, x, y, w, h):
        # Extract the region of interest (ROI) from the image
        roi = image[y:y + h, x:x + w]

        # Resize the ROI to twice its original size
        # scaled_roi = cv2.resize(roi, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        # scaled_roi = cv2.resize(roi, (0, 0), fx=1, fy=1)
        # gray_roi = cv2.cvtColor(scaled_roi, cv2.COLOR_BGR2GRAY)
        self.save_debug_image('upscale', roi)
        return roi

    def find_outside_mask(self, contour_mask_in, circle_mask):
        # Create a mask that highlights the areas where the contour mask is outside the circle mask
        outside_mask = np.zeros(contour_mask_in.shape, dtype=np.uint8)
        outside_mask[(contour_mask_in > 0) & (circle_mask == 0)] = 255
        self.save_debug_image('outside_mask', outside_mask)
        return outside_mask

    def create_clustered_image_with_curvature(self, outside_points, labels, cluster_curvatures, circle_mask_shape):
        # Create a blank canvas with the same size as the input mask
        result_image = np.zeros((*circle_mask_shape, 3), dtype=np.uint8)

        # Count the unique labels (excluding -1 for noise points)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]

        # Generate colors based on the number of unique labels
        num_colors = len(unique_labels)
        colors = generate_colors(num_colors)

        # Map labels to colors
        label_to_color = dict(zip(unique_labels, colors))

        # Draw clustered points on the canvas
        for point, label in zip(outside_points, labels):
            if label != -1:  # Ignore noise points with label -1
                x, y = point
                color = label_to_color[label]
                result_image[y, x] = color

        # Display curvature value for each cluster
        for label, curvature in cluster_curvatures.items():
            color = label_to_color[label]
            cluster_points = outside_points[labels == label]
            centroid = np.mean(cluster_points, axis=0).astype(int)
            cv2.putText(result_image, f"{curvature:.2f}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)

        # Save the result image
        self.save_debug_image("cluster_curvature_result", result_image)

    def find_extrusion_blob(self, contour_mask_in, circle_mask_in, circle_in):
        contour_mask_in = self.morph_close(contour_mask_in,3)
        outpixel_mask = self.find_outside_mask(contour_mask_in, circle_mask_in)

        self.save_debug_image('outpixel_mask', outpixel_mask)
        outpixel_mask = self.morph_close(outpixel_mask, 3)
        #self.save_debug_image('outpixel_mask_final', outpixel_mask)
        nonzero_contour = cv2.findNonZero(outpixel_mask)

        if nonzero_contour is None:
            empty_blob_image = np.zeros_like(contour_mask_in)
            return False, empty_blob_image

        # Calculate distances from circle center to each non-zero pixel
        distances = np.sqrt((nonzero_contour[:, :, 0] - circle_in[0]) ** 2 + (nonzero_contour[:, :, 1] - circle_in[1]) ** 2)

        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        mahalanobis = np.sqrt(((distances - avg_distance) / std_distance) ** 2)
        mask_mahala = mahalanobis > 1.8

        mask_combined = mask_mahala
        pixels = nonzero_contour[mask_combined]

        # Create new blank image with same shape as original image
        mahalanobis_filtered = np.zeros_like(contour_mask_in)

        # Set extracted pixels to white color (255)
        for pixel in pixels:
            x, y = pixel[0], pixel[1]
            mahalanobis_filtered[y, x] = 255

        self.save_debug_image('mahalanobis_filtered', mahalanobis_filtered)

        # Step 2: Find contours in the filtered mask
        contours, _ = cv2.findContours(mahalanobis_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 3: Filter out blobs that don't go deeper than circle_mask
        min_area = 2
        found_blob = False
        intrusion_blobs = []

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            else:
                intrusion_blobs.append(cnt)
                found_blob = True
                print("found found extrusion")

        blob_image = np.zeros_like(contour_mask_in)

        for blob in intrusion_blobs:
            cv2.drawContours(blob_image, [blob], 0, (255, 255, 255), -1)

        self.save_debug_image('extrusion_blob', blob_image)

        return found_blob, blob_image

    def create_contour_line_mask(self, mask, post_fix='contour_line_mask'):
        # Find the contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Find the outermost contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the outermost contour on a new mask

        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, 1)
        self.save_debug_image(post_fix, contour_mask)
        return contour_mask

    def find_intrusion_blob(self, contour_mask_in, circle_mask, circle_in):
        
        # Step 1: Invert the circle_mask
        inverted_circle_mask = cv2.bitwise_not(contour_mask_in)
        # Step 2: Create a combined mask with contour_mask_in and inverted_circle_mask
        combined_mask = cv2.bitwise_and(inverted_circle_mask, inverted_circle_mask, mask=circle_mask)
        self.save_debug_image('combined_mask', combined_mask)
        combined_mask = self.morph_close(combined_mask, 3)
        self.save_debug_image('combined_mask_final', combined_mask)

        nonzero_contour = cv2.findNonZero(combined_mask)

        if nonzero_contour is None:
            empty_blob_image = np.zeros_like(contour_mask_in)
            return False, empty_blob_image

        # Calculate distances from circle center to each non-zero pixel
        distances = np.sqrt((nonzero_contour[:, :, 0] - circle_in[0]) ** 2 + (nonzero_contour[:, :, 1] - circle_in[1]) ** 2)

        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        mahalanobis = np.sqrt(((distances - avg_distance) / std_distance) ** 2)
        mask_mahala = mahalanobis < 1.9
        mask_radius = distances <= (circle_in[2] * 1.0)
        mask_combined = mask_mahala & mask_radius
        pixels = nonzero_contour[mask_combined]

        # Create new blank image with same shape as original image
        mahalanobis_filtered = np.zeros_like(contour_mask_in)

        # Set extracted pixels to white color (255)
        for pixel in pixels:
            x, y = pixel[0], pixel[1]
            mahalanobis_filtered[y, x] = 255

        self.save_debug_image('mahalanobis_filtered', mahalanobis_filtered)

        # Step 2: Find contours in the mahalanobis_filtered mask
        contours, _ = cv2.findContours(mahalanobis_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 3: Filter out blobs that don't go deeper than circle_mask
        min_area = 5
        found_blob = False
        intrusion_blobs = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue

            if cv2.pointPolygonTest(cnt, (circle_mask.shape[1] // 2, circle_mask.shape[0] // 2), False) < 0:
                intrusion_blobs.append(cnt)
                found_blob = True
                print("found found intrusion")

        blob_image = np.zeros_like(contour_mask_in)

        for blob in intrusion_blobs:
            cv2.drawContours(blob_image, [blob], 0, (255, 255, 255), -1)

        self.save_debug_image('intrusion_blob', blob_image)

        return found_blob, blob_image

    def create_blob_overlay(self, image, blob_mask):
        red_overlay = np.zeros_like(image)
        red_overlay[blob_mask == 255] = (0, 0, 255)  # Red color
        return red_overlay

    def blend_images(self, image1, image2, alpha):
        blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
        return blended_image

    def save_blob_detection_result(self, original_image, blob_mask, postfix ,mask):
        color_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        original_image_cvt = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # Set the pixels corresponding to the blob to red in the original image
        color_image[blob_mask != 0] = [0, 0, 255]

        # Combine the original image and the blob pixels
        result_image = cv2.addWeighted(original_image_cvt, 0.3, color_image, 0.7, 0)

        x, y, w, h = cv2.boundingRect(mask)
        cropped_image = result_image[y:y + h, x:x + w]

        self.save_result_image(postfix, cropped_image)


if __name__ == "__main__":

    path = 'All'
    file_list = os.listdir(path)
    for file_name in file_list:
        file_path = os.path.join(path, file_name)

        if os.path.isfile(file_path):
            original_path = Path(file_path)

            lens_detector = ContactLensDetector(original_path)
            # Load the image in grayscale mode
            img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
            gradation_removed = lens_detector.remove_gradation(img, 45)
            otsu = lens_detector.otsu_threshold(gradation_removed)
            # usm = lens_detector.unsharp_mask(img,5)
            eqhist = lens_detector.eqhist(img, 'eqhist_1st')

            # rough boundary mask
            valid = cv2.bitwise_and(eqhist, otsu)
            thresholded = cv2.threshold(valid, 254, 255, cv2.THRESH_BINARY)[1]

            closed = lens_detector.morph_close(thresholded, 15)
            lens_detector.save_debug_image('rough_ring_mask', closed)

            # connect boundary edges
            contour_edge = lens_detector.fill_gaps_with_contours(closed)
            closed = lens_detector.morph_close(contour_edge, 5)

            # rough detection
            lens_area_mask, center = lens_detector.fill_from_seed(closed)
            lens_img_rough = lens_detector.filter_image_with_mask(img, lens_area_mask)

            eqhist2 = lens_detector.eqhist(lens_img_rough)
            otsu2 = lens_detector.otsu_threshold(eqhist2, postfix='otsu2')
            otsu2 = cv2.bitwise_not(otsu2)

            lens_mask_fine, circle = lens_detector.find_circle(otsu2)

            # check if thick ring
            binary_image = lens_detector.binarize(lens_img_rough, 17)
            closed = lens_detector.morph_close(binary_image, 3)
            closed = cv2.bitwise_not(closed)
            has_thick_ring = lens_detector.find_thick_ring(closed, lens_mask_fine)

            if has_thick_ring is True:
                closed_loop = lens_detector.extract_closed_loop(closed)
                found_ext_blob, ext_blob_mask = lens_detector.find_extrusion_blob(closed_loop, lens_mask_fine, circle)

                if found_ext_blob is True:
                    lens_detector.save_blob_detection_result(img, ext_blob_mask, 'extrusion_blob',lens_area_mask)

            else:
                found_ext_blob, ext_blob_mask = lens_detector.find_extrusion_blob(otsu2, lens_mask_fine, circle)
                found_int_blob, int_blob_mask = lens_detector.find_intrusion_blob(otsu2, lens_mask_fine, circle)

                found_blob = found_int_blob | found_ext_blob
                blob_bask = cv2.bitwise_or(ext_blob_mask, int_blob_mask)
                if found_blob is True:
                    lens_detector.save_blob_detection_result(img, blob_bask, 'blob',lens_area_mask)
