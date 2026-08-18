import io
import numpy as np
import cv2

def _extract_from_image_array(img_array: np.ndarray) -> np.ndarray:
    """
    Core heuristic logic to extract a 1D ECG trace from an image array.
    """
    # 1. Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array

    # 2. Thresholding: ECG traces are usually dark lines on a lighter grid background.
    # We apply inverse binary thresholding. The trace becomes white (255), background black (0).
    # Otsu's thresholding often performs well here.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Morphological operations (optional, to close gaps in the trace)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. Extract 1D array by finding the argmax or average row index of the white pixels in each column.
    height, width = thresh.shape
    signal = []

    for col in range(width):
        column_pixels = np.where(thresh[:, col] == 255)[0]
        if len(column_pixels) > 0:
            # Taking the mean position of the trace in the column
            signal.append(height - np.mean(column_pixels)) # Invert Y so up is positive
        elif len(signal) > 0:
            # Carry over last known value if gap
            signal.append(signal[-1])
        else:
            signal.append(float(height / 2)) # Default strictly halfway

    # 5. Optional smoothing and scaling
    signal_arr = np.array(signal)
    
    # Simple moving average to smooth zig-zags induced by rasterization
    window_size = max(1, width // 200)
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        signal_arr = np.convolve(signal_arr, kernel, mode='same')
    
    # Scale to typical mV range (heuristic -1.5 to 1.5)
    rng = np.ptp(signal_arr)
    if rng > 0:
        signal_arr = 3.0 * ((signal_arr - np.min(signal_arr)) / rng) - 1.5
    
    return signal_arr

def extract_signal_from_image(file_bytes: bytes) -> np.ndarray:
    """
    Reads image bytes and extracts a 1D numeric array representing the sequence.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_array is None:
        raise ValueError("Could not decode image.")
    
    return _extract_from_image_array(img_array)

def extract_signal_from_pdf(file_bytes: bytes) -> np.ndarray:
    """
    Requires PyMuPDF to convert the first page to an image.
    """
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF parsing.")

    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    if pdf_doc.page_count < 1:
        raise ValueError("PDF is empty.")
    
    # Render first page to pixmap
    page = pdf_doc.load_page(0)
    # Scale up for better resolution (matrix zoom)
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    
    # Convert PyMuPDF pixmap to numpy array
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    if pix.n == 4: # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif pix.n == 1: # GRAY
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    pdf_doc.close()
    return _extract_from_image_array(img_array)
