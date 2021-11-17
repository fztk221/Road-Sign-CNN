import cv2


def preprocessing(img):
    """Preprocessing Images by applying GrayScale->Histogram Equalization-> Normalization of pixel intensities"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    img = cv2.equalizeHist(img)  # standardize the lighting
    img = img / 255  # normalize values between 0 and 1 rather than 0 to 255
    return img
