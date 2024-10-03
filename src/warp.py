import matplotlib.pyplot as plt
import cv2
import numpy as np
from rembg import remove
from addons import sort_points, show_image_with_points, load_and_process_image, calculate_exceeding_area, clean_lines

def apply_filter(image, plot=False):
    """
    Define a 5x5 kernel and apply the filter to a grayscale image.

    Args:
        image (np.array): The input image in RGB format.
        plot (bool): Whether to display the plot of the filtered image.

    Returns:
        np.array: The filtered image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    if plot:
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title('Filtered Image')
        plt.show()
    return filtered

def apply_threshold(filtered, plot=False):
    """
    Apply threshold to the filtered image.

    Args:
        filtered (np.array): The filtered image.
        plot (bool): Whether to display the plot of the thresholded image.

    Returns:
        np.array: The thresholded image.
    """
    ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_TRIANGLE)
    if plot:
        plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
        plt.title('After applying TRIANGLE threshold')
        plt.show()
    return thresh

def detect_contour(img, image_shape, plot=False):
    """
    Detect the largest contour in the thresholded image.

    Args:
        img (np.array): The thresholded image.
        image_shape (tuple): The shape of the original image.
        plot (bool): Whether to display the plot of the largest contour.

    Returns:
        np.array: The canvas with the largest contour drawn.
        list: The largest contour detected.
    """
    canvas = np.zeros(image_shape, np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)
    if plot:
        plt.title('Largest Contour')
        plt.imshow(canvas)
        plt.show()
    return canvas, cnt

def find_enclosing_quadrilateral(canvas, cnt, plot=False):
    """
    Find and draw an enclosing quadrilateral and bounding box around the largest contour.

    Args:
        canvas (np.array): The canvas to draw the enclosing quadrilateral and bounding box.
        cnt (list): The largest contour detected.
        plot (bool): Whether to display the plot of the enclosing quadrilateral.

    Returns:
        np.array: The approximated quadrilateral points with exactly 4 corners.
        tuple: Coordinates of the bounding box (x, y, w, h).
    """
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    while len(approx) != 4:
        if len(approx) > 4:
            epsilon += 0.01 * cv2.arcLength(cnt, True)
        else:
            epsilon -= 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(canvas, [approx], -1, (0, 255, 0), 3)

    if plot:
        plt.title('Enclosing Quadrilateral')
        plt.imshow(canvas)
        plt.show()

    return approx

def warp_vertex_by_vertex(image, contour, approx, bounding_box, threshold=10, steps=20, plot=False):
    """
    Gradually warp the image, moving each vertex individually towards the target step by step.

    Args:
        image (np.array): The original image to be warped.
        contour (np.array): The original contour points.
        approx (np.array): The quadrilateral points to be warped.
        bounding_box (tuple): (x, y, w, h) coordinates of the bounding box.
        threshold (int): The maximum allowable distance outside the bounding box for warping.
        steps (int): The number of gradual steps for each vertex.
        plot (bool): Whether to display the plots during the warping process.

    Returns:
        np.array: The warped version of the image.
    """
    
    contour = np.array(contour, dtype=np.float32).reshape(-1, 1, 2)  # Ensure correct shape
    x, y, w, h = bounding_box
    bbox_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

    if len(approx) != 4:
        raise ValueError("The approx shape must have exactly 4 points.")

    src_points = np.array([point[0] for point in approx], dtype=np.float32)
    src_points = sort_points(src_points)  # Ensure points are in a consistent order
    dst_points = sort_points(bbox_corners)

    if plot:
        show_image_with_points(image, src_points, "Initial Points", dst_points, contour=contour)

    warped_image = image.copy()

    for vertex_index in range(4):
        destination = src_points.copy()  # Copy the initial source points to modify for this vertex
        stop_flag = False  # Reset stop flag for each vertex

        for step in range(steps):
            if not stop_flag:
                # Calculate the difference and euclidean distance between the current vertex and the target
                diff = calculate_exceeding_area(contour, bounding_box, image.shape[:2])
                euclidean_diff = np.linalg.norm(destination[vertex_index] - dst_points[vertex_index])

                # Check stopping condition
                if diff > (threshold * (vertex_index + 1) / 2) or euclidean_diff < 1:  # Adjust threshold for finer control
                    stop_flag = True
                else:
                    destination[vertex_index] += (dst_points[vertex_index] - destination[vertex_index])

                # Calculate the transformation matrix with updated destination
                matrix = cv2.getPerspectiveTransform(src_points, destination)

                # Apply perspective transform on the contour (for the next iteration)
                contour = cv2.perspectiveTransform(contour.reshape(-1, 1, 2), matrix).reshape(-1, 2)

                # Apply the transformation on the image
                image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

        if plot:
            show_image_with_points(image, src_points, f"Vertex {vertex_index + 1}", destination, dst_points, contour)

        # After steps are done for this vertex, update the source points for the next iteration
        src_points = destination

    return image

import time

def process_image_recursively(image, num_recursions, plot=False, threshold=10, steps=20):
    """
    Process the image recursively using the defined pipeline functions, measuring the time of each recursion.

    Args:
        image (np.array): The input image.
        num_recursions (int): The total number of recursive applications.
        plot (bool): Whether to display intermediate plots.

    Returns:
        np.array: The final processed image after all recursions.
    """
    
    filtered = apply_filter(image, plot=plot)
    thresh = apply_threshold(filtered, plot=plot)
    canvas, cnt = detect_contour(thresh, image.shape, plot=plot)
        
    x, y, w, h = cv2.boundingRect(cnt)
    bounding_box = (x, y, w, h)

    # Apply the image processing steps
    for i in range(num_recursions):
        start_time = time.time()  # Start timing the iteration
        
        filtered = apply_filter(image, plot=plot)
        thresh = apply_threshold(filtered, plot=plot)
        canvas, cnt = detect_contour(thresh, image.shape, plot=plot)
        
        approx = find_enclosing_quadrilateral(canvas, cnt, plot=plot)
        image = warp_vertex_by_vertex(image, cnt, approx, bounding_box, threshold=threshold, steps=steps, plot=plot)
        
        end_time = time.time()  # End timing the iteration
        iteration_time = end_time - start_time  # Calculate the duration of the iteration
        print(f"Processing iteration {i + 1}/{num_recursions} took {iteration_time:.4f} seconds")
    
    # Draw the bounding box on the final image
    final_image_with_box = image.copy()
    cv2.rectangle(final_image_with_box, (x, y), (x + w, y + h), (0, 0, 255), 4)  # Blue bounding box

    return final_image_with_box


# İki çizginin kesişim noktalarını hesaplayan fonksiyon
def find_intersection(line1, line2, height, width):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Determinant hesaplama
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # Çizgiler paralel, kesişim yok
    
    # Kesişim noktalarını hesaplama
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    if px < 0 or px > width or py < 0 or py > height:
        return None
    return int(px), int(py)

def hough_lines(image, lines, height, width, cnt, output_path):
    rho_threshold = 50  # Mesafe eşik değeri
    theta_threshold = np.pi / 180 * 30
    
    # Benzer çizgileri birleştir
    merged_lines = clean_lines(lines, rho_threshold, theta_threshold)
    
    x, y, w, h = cv2.boundingRect(cnt)
        
    # Çizgileri çiz
    def is_point_on_boundary(point, boundary_lines):
        for line in boundary_lines:
            x1, y1 = line[0]
            x2, y2 = line[1]
            a1, b1 = point
            
            if abs(a1-x1) < 5 and abs(b1-y1) < 5:
                return True
            elif abs(a1-x2) < 5 and abs(b1-y2) < 5:
                return True
            
        return False

    my_lines = []
    for i, r_theta in enumerate(merged_lines):
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        
        # NaN kontrolü yapın
        if np.isnan(r) or np.isnan(theta):
            continue
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        
        # NaN kontrolü yapın
        if np.isnan(x0) or np.isnan(y0) or np.isnan(a) or np.isnan(b):
            continue
        
        mean = 3 * (height + width) / 4
        
        x1 = int(x0 + mean * (-b))
        y1 = int(y0 + mean * (a))
        x2 = int(x0 - mean * (-b))
        y2 = int(y0 - mean * (a))
        
        my_lines.append(((x1, y1), (x2, y2)))
        
    left_up = (x, y)
    right_up = (x+w, y)
    left_down = (x, y+h)
    right_down = (x+w, y+h)

    boundary_lines = []
    boundary_lines.append((left_up, right_up))      # Üst çizgi
    boundary_lines.append((right_up, right_down))   # Sağ çizgi
    boundary_lines.append((right_down, left_down))  # Alt çizgi
    boundary_lines.append((left_down, left_up))     # Sol çizgi
    
    my_lines.append((left_up, right_up))
    my_lines.append((right_up, right_down))
    my_lines.append((right_down, left_down))
    my_lines.append((left_down, left_up))

    intersections = []
    for i in range(len(my_lines)):
        for j in range(i+1, len(my_lines)):
            intersect = find_intersection(my_lines[i][0] + my_lines[i][1], my_lines[j][0] + my_lines[j][1], height, width)
            if intersect is not None and not is_point_on_boundary(intersect, boundary_lines):
                intersections.append(intersect)

    def filter_close_points(points, threshold=10):
        filtered_points = []
        
        for i, point in enumerate(points):
            keep = True  # Bu noktayı tutup tutmama kontrolü
            for other_point in filtered_points:
                # Tuple'ları numpy array'e çevir ve mesafeyi hesapla
                if abs(np.linalg.norm(np.array(point) - np.array(other_point))) < threshold:
                    keep = False
                    break  # Eğer yakınsa, bu noktayı eklemeyi bırak ve diğerine geç
            if keep:
                filtered_points.append(point)  # Nokta yeterince uzaktaysa ekle
        
        return filtered_points

    # Yakın olan noktaları filtrele (mesafe eşik değeri 10 pixel)
    filtered_points = filter_close_points(intersections, threshold=200)
    
    boundary_points = [left_up, right_up, right_down, left_down]
    
    if len(filtered_points) == 4:
        source_np = np.array(filtered_points, dtype=np.float32)
        source_np = sort_points(source_np)
        
        dest_np = np.array(boundary_points, dtype=np.float32)
        dest_np = sort_points(dest_np)
        
        # Perspectif dönüşüm matrisi hesapla
        matrix = cv2.getPerspectiveTransform(source_np, dest_np)

        # Dönüşümü görüntüye uygula
        image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

        for point in filtered_points:
            cv2.circle(image, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)  # Kırmızı noktalarla kesişim yerlerini işaretle

        for line in boundary_lines:
            cv2.line(image, line[0], line[1], (0, 255, 0), 2)  # Yeşil çizgilerle çizgileri çiz

        # Çizgilerin olduğu resmi kaydet
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return True
    else:
        return False
    
def warp(image_path, output_path):
    # Load and optionally remove the background from the image
    image = load_and_process_image(image_path, remove_background=True)

    plot = False

    filtered = apply_filter(image, plot=plot)
    thresh = apply_threshold(filtered, plot=plot)
    canvas, cnt = detect_contour(thresh, image.shape, plot=plot)

    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    height, width = image.shape[:2]

    if lines is not None:
        if not hough_lines(image, lines, height, width, cnt, output_path):
            # Apply the processing recursively 10 times
            final_image = process_image_recursively(image, num_recursions=10, plot=False, threshold=3, steps=20)

            # Save the final warped image to a file
            cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
