import matplotlib.pyplot as plt
import cv2
import numpy as np
from rembg import remove

def sort_points(points):
    """
    Sort points in the order: top-left, top-right, bottom-right, bottom-left.

    Args:
        points: np.array of shape (4, 2), points to be sorted.

    Returns:
        sorted_points: np.array, sorted points in the specified order.
    """
    # Sort by y-coordinate (top to bottom), then by x-coordinate (left to right)
    sorted_y = points[np.argsort(points[:, 1])]
    top_points = sorted_y[:2]
    bottom_points = sorted_y[2:]

    # Sort left to right
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def show_image_with_points(image, points, title, second_points=None, third_points=None, contour=None):
    """
    Display the image with given points overlaid, optionally showing target points.

    Args:
        image: np.array, the image to display.
        points: np.array, the points to plot on the image (source points).
        title: str, the title of the plot.
        second_points: np.array, optional, the second set of points to plot on the image.
        third_points: np.array, optional, the third set of points to plot on the image.
        contour: np.array, optional, the contour to draw on the image.
    """
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], color='red', marker='o', label='Source Points')  # Mark the source points
    
    if contour is not None:
        # Ensure the contour has valid points and correct shape
        if len(contour) > 0 and isinstance(contour, np.ndarray):
            contour = contour.reshape(-1, 1, 2).astype(np.int32)  # Reshape to correct format
            canvas = np.zeros_like(image)
            # Draw the contour on the canvas
            cv2.drawContours(canvas, [contour], -1, (0, 255, 255), 3)
            # Overlay the canvas on the existing plot with contours
            plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), alpha=0.5)
        else:
            print("Warning: Contour is empty or not properly formatted.")

    # If second points are provided, plot them as well
    if second_points is not None:
        plt.scatter(second_points[:, 0], second_points[:, 1], color='lightblue', marker='x', label='Second Points')
    
    if third_points is not None:
        plt.scatter(third_points[:, 0], third_points[:, 1], color='lightgreen', marker='x', label='Third Points')

    plt.title(title)
    plt.legend()
    plt.show()

    
    
def load_and_process_image(image_path, remove_background=False):
    """
    Load an image, optionally remove its background, convert it to RGB format, and add 20% padding around it.

    Args:
        image_path (str): The path to the image file.
        remove_background (bool): Whether to remove the background of the image.

    Returns:
        np.array: The processed image in RGB format with padding.
    """
    # Load the image
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    # Optionally remove the background
    if remove_background:
        image_data = remove(image_data)

    # Convert the resulting image data to a NumPy array
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Convert the image to RGB if it has an alpha channel
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate padding size (20% of width and height)
    height, width, _ = image.shape
    pad_height = int(height * 0.2)
    pad_width = int(width * 0.2)

    # Add padding to the image
    padded_image = cv2.copyMakeBorder(
        image,
        top=pad_height, bottom=pad_height,
        left=pad_width, right=pad_width,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # White padding, you can change the color
    )

    return padded_image


def calculate_exceeding_area(warped_cnt, bounding_box, image_shape, plot=False):
    """
    Calculate the percentage of the area where the warped contour exceeds the bounding box.

    Args:
        warped_cnt (np.array): The warped contour points.
        bounding_box (tuple): The coordinates of the bounding box (x, y, w, h).
        image_shape (tuple): The shape of the image.

    Returns:
        exceeding_percentage (float): The percentage of the area in pixels where the contour exceeds the bounding box.
    """
    
    # Ensure the contour is in the correct shape and has points
    if len(warped_cnt) == 0 or not isinstance(warped_cnt, np.ndarray):
        print("Error: warped_cnt is empty or not a valid contour.")
        return 0.0

    # Ensure warped_cnt is in the correct shape (n, 1, 2)
    warped_cnt = np.array(warped_cnt, dtype=np.int32).reshape((-1, 1, 2))

    # Create empty masks for the bounding box and the warped contour
    bounding_box_mask = np.zeros(image_shape, np.uint8)
    warped_contour_mask = np.zeros(image_shape, np.uint8)

    # Draw the bounding box on its mask
    x, y, w, h = bounding_box
    cv2.rectangle(bounding_box_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)

    # Draw the warped contour on its mask
    if len(warped_cnt) > 0:
        cv2.drawContours(warped_contour_mask, [warped_cnt], -1, 255, thickness=cv2.FILLED)

    # Calculate the difference between the contour and the bounding box
    exceeding_mask = cv2.subtract(warped_contour_mask, bounding_box_mask)

    # Count the number of pixels where the contour exceeds the bounding box
    exceeding_area = cv2.countNonZero(exceeding_mask)

    # Calculate the total area of the bounding box
    bounding_box_area = cv2.countNonZero(bounding_box_mask)

    # Calculate the percentage of the exceeding area relative to the bounding box area
    if bounding_box_area == 0:
        print("Error: Bounding box area is zero, cannot calculate percentage.")
        exceeding_percentage = 0.0
    else:
        exceeding_percentage = (exceeding_area / bounding_box_area) * 100

    # Combine the masks into a single color plot
    combined_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    combined_image[bounding_box_mask > 0] = [0, 255, 0]  # Green for the bounding box
    combined_image[warped_contour_mask > 0] = [255, 0, 0]  # Blue for the warped contour
    combined_image[exceeding_mask > 0] = [255, 0, 255]  # Magenta for the exceeding area

    if plot:
        # Display the combined plot
        plt.figure(figsize=(10, 10))
        plt.imshow(combined_image)
        plt.title('Bounding Box (Green), Warped Contour (Blue), Exceeding Area (Magenta)')
        plt.axis('off')
        plt.show()

    return exceeding_percentage

def clean_lines(lines, rho_threshold, theta_threshold):
    # Benzer çizgileri gruplayarak ortalama çizgiler elde eder
    merged_lines = []
    
    for i in range(len(lines)):
        r1, theta1 = lines[i][0]
        if np.isnan(r1) or np.isnan(theta1):
            continue
        
        for j in range(i + 1, len(lines)):
            r2, theta2 = lines[j][0]
            if np.isnan(r2) or np.isnan(theta2):
                continue
            
            # Eğer iki çizgi birbirine yakınsa, onları birleştir
            if abs(r1 - r2) < rho_threshold and abs(theta1 - theta2) < theta_threshold:
                # Ortalama çizgi oluşturma
                lines[i][0] = [(r1 + r2) / 2, (theta1 + theta2) / 2]
                lines[j][0] = [np.nan, np.nan]  # İkinci çizgiyi iptal et
    
    # NaN olmayan çizgileri filtrele
    for line in lines:
        if not np.isnan(line[0][0]) and not np.isnan(line[0][1]):
            merged_lines.append(line)
    
    return merged_lines

def is_line_similar(line, existing_lines, rho_threshold, theta_threshold):
    # Mevcut çizgilerle karşılaştırıp benzer olup olmadığını kontrol eder
    rho, theta = line
    for r, t in existing_lines[0]:
        if abs(rho - r) < rho_threshold and abs(theta - t) < theta_threshold*3:
            return True  # Benzer çizgi zaten mevcut
    return False  # Benzer çizgi yok