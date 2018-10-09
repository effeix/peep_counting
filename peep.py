########################################################################
# IMPORTS                                                              #
########################################################################
import pickle
import timeit
from datetime import timedelta
from os import listdir, environ
from os.path import abspath, dirname, join
from random import choice, shuffle
from sys import argv

import cv2
import matplotlib.pyplot as plt
import numpy as np


########################################################################
# PARAMETERS                                                           #
########################################################################
EPSILON_1  = 12
EPSILON_2  = 8
ALPHA      = 0.4
BETA       = 1.1
N_IMAGES   = int(environ.get("N", None))
TRAIN      = int(environ["TR"])
TEST_IMAGE = environ.get("TEST_IMAGE", None)


########################################################################
# HELPERS                                                              #
########################################################################
def random_images(n=10, d="BG", v=1):
    """Generates a list of paths to random images.

    The images are located in the ìmages/ directory, divided in 3 datasets:
        - BG (Background): images intended for training
        - CC (City Center): images with fewer people
        - L1 (Large Group): images with more people

    Each section contains pictures of up to 4 cameras with different angles,
    taken at several moments of the day.

    Args:
        n: number of random images to gather
        d: dataset ("BG", "CC" or "L1")
        v: camera (view) to be used

    Returns:
        List of strings representing relative paths to images/ directory

    Raises:
        AssertionError: v must be between 0 (excluded) and 4 (included)
    """
    assert (0 < v <= 4)

    base_path = f"../images/{d}/{v}/"
    timestamps = listdir(base_path) 

    images = []
    for _ in range(n):
        img_dir = join(base_path, choice(timestamps))
        img_path = choice(listdir(img_dir))
        images.append(join(img_dir, img_path))

    return images, d, v

def read_img(file):
    """Reads image from specific file and creates a Numpy array.

    The picture must be converted to RGB because OpenCV defaults to BGR.
    Pixel data is converted to double for calculation sake.

    Args:
        file: string containing path to image

    Returns:
        RGB image with pixel data as double
    """
    i = cv2.cvtColor(
        cv2.imread(file),
        cv2.COLOR_BGR2RGB
    ).astype(float)

    return i

def count_cw(codebooks):
    """Counts codewords in model.
    """
    return sum([len(i) for i in codebooks])


########################################################################
# CODEWORD RELATED                                                     #
########################################################################
def brightness(I, Imin, Imax, alpha, beta):
    """Detect if pixel brightness is inside brightness range.

    Args:
        I: pixel brightness
        Imin: min allowed brightness
        Imax: max allowed brightness
        alpha: hyperparameter
        beta: hyperparameter

    Returns:
        True if brightness is inside range.
        False otherwise.
    """
    return alpha*Imax <= I <= min(beta*Imax, Imin/alpha)

def colordist(X, V):
    """Calculates color distortion between to pixels

    If the pixel is black, pixel module will be zero, causing a
    FloatingPointError in the calculation. Thus, if module is zero,
    p_squared is zeroed automatically.

    Args:
        X, V: rgb arrays representing one pixel each

    Returns:
        Color distortion 
    """
    mod_X = sum([i**2 for i in X])
    mod_V = sum([i**2 for i in V])
    inner_prod_squared = sum([X[i]*V[i] for i in range(len(V))])**2
    
    if mod_V == 0.0:
        p_squared = 0.0
    else:
        p_squared = inner_prod_squared / mod_V
    
    return abs(mod_X - p_squared)**0.5

def create_codeword(cb, rgb, I, t):
    """Creates new codeword.

    Args:
        cb: codebooks array
        rgb: pixel vector
        I: pixel brightness
        t: current iteration

    """
    cw = list(rgb) + [I, I, 1, t-1, t, t]
    cb.append(cw)

def update_codeword(cw, rgb, I, t):
    """Update codeword with new values.

    Args:
        cw: codeword
        rgb: pixel vector
        I: pixel brightness
        t: current iteration
    """
    cw[0] = (cw[5] * cw[0] + rgb[0]) / (cw[5] + 1)
    cw[1] = (cw[5] * cw[1] + rgb[1]) / (cw[5] + 1)
    cw[2] = (cw[5] * cw[2] + rgb[2]) / (cw[5] + 1)
    cw[3] = min(I, cw[3])
    cw[4] = max(I, cw[4])
    cw[5] += 1
    cw[6] = max(cw[6], t - cw[8])
    cw[8] = t

def match(X, V, I, Imin, Imax, alpha, beta, epsilon):
    """Checks if two pixels matched based on colordist and brightness.

    Args:
        (Explained above)

    Returns:
        True if matched.
        False otherwise.
    """
    return colordist(X, V) < epsilon and brightness(I, Imin, Imax, alpha, beta)


########################################################################
# TRAINING PHASE                                                       #
########################################################################
def train(imgs):
    
    # Iterate over N images
    for t, file in enumerate(images, start=1):

        img = read_img(file)

        # If it is the first iteration, create the codebooks
        if t == 1:
            n_codebooks = img.shape[0] * img.shape[1]
            codebooks = [[] for _ in range(n_codebooks)]

        start_time = timeit.default_timer()

        # For each pixel (codebook)
        idx = -1
        for px in np.ndindex(img.shape[:2]):
            idx += 1

            # Calculate pixel brightness
            I = (img[px][0]**2 + img[px][1]**2 + img[px][2]**2)**0.5

            codebook_empty = True
            no_match = True

            # For each codeword inside the current codebook
            for cw in codebooks[idx]:
                codebook_empty = False

                # If pixel in current image matched with existing codeword
                # based on colordist and brightness
                if match(img[px], cw[0:3], I, cw[3], cw[4],
                        ALPHA, BETA, EPSILON_1):

                    # Update codeword as in Section 2 of the article
                    update_codeword(cw, img[px], I, t)
                    
                    no_match = False

                    break

            # If there are no codewords in current codebook or there is no
            # matching codeword, create a new codeword with current pixel info
            if codebook_empty or no_match:
                create_codeword(codebooks[idx], img[px], I, t)

        # Verbose
        number_fill = len(str(len(images)))
        filename = "/".join(file.split("/")[-2:])
        print(f"- IMG {t:0{number_fill+1}d} ({filename}) took ", end="")
        print(f"{timeit.default_timer() - start_time:.3f} seconds")

    # Update maximum negative run-length
    print("\nUPDATING LAMBDAS")
    for codebook in codebooks:
        for codeword in codebook:
            codeword[6] = max(codeword[6], (len(images)-codeword[8]+codeword[7]-1))

    for codebook in codebooks:
        for codeword in codebook:
            if codeword[6] > N_IMAGES / 2:
                # I can't delete right after finding because it would mess
                # up the for loop
                codeword = None

        codebook = [cw for cw in codebook if cw is not None]


    print(f"Codewords: {count_cw(codebooks)} (~{count_cw(codebooks)/n_codebooks:.2f}/codebook)")

    print("\nSAVING TRAINING DATA TO train.pickle\n")
    with open('train.pickle', 'wb') as f:
        pickle.dump(codebooks, f, protocol=pickle.HIGHEST_PROTOCOL)


########################################################################
# TEST PHASE                                                           #
########################################################################
def subtraction(codebooks, img):
    bw_img = np.copy(img)

    idx = -1
    for px in np.ndindex(img.shape[:2]):
        idx += 1

        I = (img[px][0]**2 + img[px][1]**2 + img[px][2]**2)**0.5

        for cw in codebooks[idx]:

            if match(img[px], cw[0:3], I, cw[3], cw[4], ALPHA, BETA, EPSILON_2):
                update_codeword(cw, img[px], I, 1)
                bw_img[px] = [0, 0, 0]
                break

            bw_img[px] = [255, 255, 255]

    return bw_img, img


########################################################################
# HELPERS                                                              #
########################################################################
def count_people(subtracted, original):
    # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    # https://makehardware.com/2016/05/19/blob-detection-with-python-and-opencv/
    k = np.ones((7, 7), dtype=np.uint8)
    eroded = cv2.erode(subtracted, k, iterations=1)
    dilated = cv2.dilate(eroded, k, iterations=1)
    
    plt.imshow(dilated)
    plt.show()

    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByInertia = False
    detector_params.filterByConvexity = False
    detector_params.filterByArea = True
    detector_params.minArea = 500

    detector = cv2.SimpleBlobDetector_create(detector_params)
    rev = 255 - np.uint8(dilated)
    
    keypoints = detector.detect(rev)
    
    im_with_keypoints = cv2.drawKeypoints(
        np.uint8(original),
        keypoints,
        np.array([]),
        (255,0,0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.imshow(im_with_keypoints)
    plt.show()

    return len(keypoints)

if __name__ == '__main__':

    if TRAIN == 1:
        images, d, v = random_images(N_IMAGES)
        print(f"\nTRAINING ({d}/View {v})")
        train(images)

    elif TRAIN == 0:
        if TEST_IMAGE is None:
            exit("No test image!")
        
        with open("train.pickle", "rb") as f:
            codebooks = pickle.load(f)

        print("TESTING")
        img = read_img(TEST_IMAGE)

        start_time = timeit.default_timer()

        subtracted, original = subtraction(codebooks, img)

        print(f"- IMG ({TEST_IMAGE}) took {timeit.default_timer() - start_time:.3f} seconds")

        n_people = count_people(subtracted, original)

        print(f"People count: {n_people}")
