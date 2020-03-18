# -*- coding: utf-8 -*-
import matplotlib
import cv2
import sys
import copy
import collections
import math
import string
import os
import numpy as np
from math import log
from matplotlib import pyplot as plt
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from huffman import*


def zig_zag(input_matrix, block_size):
    z = np.empty([block_size*block_size])
    index = -1
    bound = 0
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                z[index] = input_matrix[j, i-j]
            else:
                z[index] = input_matrix[i-j, j]
    return z


def zig_zag_reverse(input_matrix):
    block_size = 8
    output_matrix = np.empty([block_size, block_size])
    index = -1
    bound = 0
    input_m = []
    for i in range(0, 2 * block_size - 1):
        if i < block_size:
            bound = 0
        else:
            bound = i - block_size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                output_matrix[j, i - j] = input_matrix[0][index]
            else:
                output_matrix[i - j, j] = input_matrix[0][index]
    return output_matrix


def MSE(img1, img2):
    return ((img1.astype(np.float) - img2.astype(np.float)) ** 2).mean(axis=None)


def PSNR(mse): 
    return 10 * log(((255 * 255) / mse), 10)


def SSIM(img1, img2):
    return ssim(img1.astype(np.float), img2.astype(np.float), data_range=img2.max() - img2.min())


def Compression_Ratio(filepath):
    fzOne = os.stat(filepath).st_size
    fzOne = fzOne/1024
    fzTwo = os.path.getsize('data/decompressed.tif')
    fzTwo = fzTwo/1024
    icr = fzOne/float(fzTwo)
    return icr


def main():
    filepath = sys.argv[1] # jpg, tif or png
    multiplying_factor = float(sys.argv[2])
    img = cv2.imread(filepath, 0)
    # quantization table
    qtable = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]])
    # multiply quantization table
    for i in range(0, len(qtable)):
        for j in range(0, len(qtable)):
            qtable[i, j] = qtable[i, j] * multiplying_factor

    ################## JPEG compression ##################
    iHeight, iWidth = img.shape[:2]
    zigZag = []
    for startY in range(0, iHeight, 8):
        for startX in range(0, iWidth, 8):
            block = img[startY:startY+8, startX:startX+8]
            # apply DCT for a block
            blockf = np.float32(block)  # float conversion
            dct = cv2.dct(blockf)  # dct
            # quantization of the DCT coefficients
            blockq = np.floor(np.divide(dct, qtable)+0.5)
            # Zig Zag
            zigZag.append(zig_zag(blockq, 8))
    # DPCM for DC value
    dc = []
    dc.append(zigZag[0][0])  # first value stays the same
    for i in range(1, len(zigZag)):
        dc.append(zigZag[i][0]-zigZag[i-1][0])
    # RLC for AC values
    rlc = []
    zeros = 0
    for i in range(0, len(zigZag)):
        zeros = 0
        for j in range(1, len(zigZag[i])):
            if (zigZag[i][j] == 0):
                zeros += 1
            else:
                rlc.append(zeros)
                rlc.append(zigZag[i][j])
                zeros = 0
        if(zeros != 0):
            rlc.append(zeros)
            rlc.append(0)

    #### Huffman ####
    # Huffman DPCM
    # Find frequency of appearance for each value of the list
    counterDPCM = collections.Counter(dc)
    # Define probs list as list of pairs (Unique item, Corresponding frequency)
    probsDPCM = []
    for key, value in counterDPCM.items():
        probsDPCM.append((key, np.float32(value)))
    # Creates a list of nodes ready for the Huffman algorithm 'iterate'.
    symbolsDPCM = makenodes(probsDPCM)
    # runs the Huffman algorithm on a list of "nodes". It returns a pointer to the root of a new tree of "internalnodes".
    rootDPCM = iterate(symbolsDPCM)
    # Encodes a list of source symbols.
    sDPMC = encode(dc, symbolsDPCM)
    # Huffman RLC
    # Find frequency of appearance for each value of the list
    counterRLC = collections.Counter(rlc)
    # Define probs list as list of pairs (Unique item, Corresponding frequency)
    probsRLC = []
    for key, value in counterRLC.items():
        probsRLC.append((key, np.float32(value)))
    # Creates a list of nodes ready for the Huffman algorithm 'iterate'.
    symbolsRLC = makenodes(probsRLC)
    # runs the Huffman algorithm on a list of "nodes". It returns a pointer to the root of a new tree of "internalnodes".
    root = iterate(symbolsRLC)
    # Encodes a list of source symbols.
    sRLC = encode(rlc, symbolsRLC)

    ################## JPEG decompression ##################

    #### Huffman ####
    # Huffman DPCM
    # Decodes a binary string using the Huffman tree accessed via root
    dDPCM = decode(sDPMC, rootDPCM)
    decodeDPMC = []
    for i in range(0, len(dDPCM)):
        decodeDPMC.append(float(dDPCM[i]))
    # Huffman RLC
    # Decodes a binary string using the Huffman tree accessed via root
    dRLC = decode(sRLC, root)
    decodeRLC = []
    for i in range(0, len(dRLC)):
        decodeRLC.append(float(dRLC[i]))
    # Inverse DPCM
    inverse_DPCM = []
    inverse_DPCM.append(decodeDPMC[0])  # first value stays the same
    for i in range(1, len(decodeDPMC)):
        inverse_DPCM.append(decodeDPMC[i] + inverse_DPCM[i-1])
    # Inverse RLC
    inverse_RLC = []
    for i in range(0, len(decodeRLC)):
        if (i % 2 == 0):
            if(decodeRLC[i] != 0.0):
                if(i+1 < len(decodeRLC) and decodeRLC[i+1] == 0):
                    for j in range(1, int(decodeRLC[i])):
                        inverse_RLC.append(0.0)
                else:
                    for j in range(0, int(decodeRLC[i])):
                        inverse_RLC.append(0.0)
        else:
            inverse_RLC.append(decodeRLC[i])
    new_img = np.empty(shape=(iHeight, iWidth))
    height = 0
    width = 0
    temp = []
    temp2 = []
    for i in range(0, len(inverse_DPCM)):
        temp.append(inverse_DPCM[i])
        for j in range(0, 63):
            temp.append((inverse_RLC[j+i*63]))
        temp2.append(temp)
        # inverse zig-zag and inverse Quantization of the DCT coefficients
        inverse_blockq = np.multiply(np.reshape(
            zig_zag_reverse(temp2), (8, 8)), qtable)
        # inverse dct
        inverse_dct = cv2.idct(inverse_blockq)
        for startY in range(height, height+8, 8):
            for startX in range(width, width+8, 8):
                new_img[startY:startY+8, startX:startX+8] = inverse_dct
        width = width + 8
        if(width == iHeight):
            width = 0
            height = height + 8
        temp = []
        temp2 = []
    np.place(new_img, new_img > 255, 255)  # saturation
    np.place(new_img, new_img < 0, 0)  # grounding
    # display the images
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_img, cmap='gray'), plt.title(
        'Image after Jpeg decompression')
    plt.xticks([]), plt.yticks([])
    plt.show()
    # save the decompressed image
    cv2.imwrite('data/decompressed.tif', new_img)
    # MSE
    mse = MSE(img, new_img)
    print("MSE = ", mse)
    # PSNR
    print("PSNR = ", PSNR(mse))
    # calculate SSIM
    print("SSIM = ", SSIM(img, new_img))
    # Compression Ratio
    print("Compression Ratio = ", Compression_Ratio(filepath))


if __name__ == "__main__":
    main()
