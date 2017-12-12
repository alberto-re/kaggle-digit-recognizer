"""
Dataset image viewer utility
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt

PIXEL_MID_VALUE = 127

def invert(pixel):
    """
    Invert colors
    """
    if pixel < PIXEL_MID_VALUE:
        pixel = PIXEL_MID_VALUE + (PIXEL_MID_VALUE - pixel)
    else:
        pixel = PIXEL_MID_VALUE - (pixel - PIXEL_MID_VALUE)
    return pixel


def main(argv):
    """
    Application logic entry point
    """
    dataset = argv[0]
    rown = int(argv[1])

    df = pd.read_csv("./data/%s.csv" % dataset, header=0, sep=",", encoding="utf-8")

    if dataset == "train":
        df.drop(["label"], axis=1, inplace=True)
 
    # Extract the Numpy array representing the chosen digit
    img = df.iloc[[rown]].values

    # Invert image color
    img[0] = [invert(px) for px in img[0]]

    # Reshape the array from [1, 784] to [28, 28]
    img = img.reshape((28, 28))

    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
