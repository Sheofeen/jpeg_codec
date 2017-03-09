import cv2
import numpy as np
from math import sqrt, cos, pi
import sys

HextoOct = lambda x: int(x, 16)
OcttoHex = lambda x: hex(x)

def readData(filepath):
	fin = open(filepath, "rb")
	data = fin.read()
	fin.close()

	pixels = map(HextoOct, data.strip().split()) 
	#pixels = data.strip().split()  # test data
	return pixels

# transform from array to matrix
def arrayToMat(data, width, height):
	img = np.zeros((height, width), dtype="uint8")
	ctr = 0
	for i in xrange(height):
		for j in xrange(width):
			img[i][j] = data[ctr]
			ctr += 1
	return img

# divide the image into blocks of size 8 * 8
def divideIntoBlock(dataMat, blocksize = 8):
	blocks = []
	height, width = dataMat.shape
	# pending
	newHeight = ((height / blocksize) + min(1, height % blocksize)) * blocksize
	newWidth = ((width / blocksize) + min(1, width % blocksize)) * blocksize
	newData = np.zeros((newHeight, newWidth), dtype="uint8")
	newData[:height, :width] = dataMat
	# right expand
	for i in xrange(newWidth - width):
		newData[:height, width + i: width + i + 1] = dataMat[:, width - 1]
	# bottom expand
	for i in xrange(newHeight - height):
		newData[height + i,:] = newData[height - 1,:]

	for i in range(0, newHeight, blocksize):
		for j in range(0, newWidth, blocksize):
			blocks.append(np.mat(newData[i: i + blocksize, j: j+blocksize], dtype = "float"))
	return blocks

# check the block result
def checkBlocks(blocks):
	fout = open("test.txt", "w")
	for i in xrange(len(blocks)):
		fout.write("block %d\n" % i)
		for row in blocks[i]:
			for elem in row:
				fout.write(hex(elem) + " ")
			fout.write("\n")
	fout.close()

# normal DCT using matrix multiplication
def DCT(data, transformaMat, transformaMat_inv):
	return transformaMat * (data - 128) * transformaMat_inv
	#return cv2.dct(data - 128)  # for result checking

# normal IDCT using matrix multiplication
def IDCT(spectrum, transformaMat, transformaMat_inv):
	return np.mat((transformaMat_inv * spectrum * transformaMat) + 128, dtype="uint8")
	#return cv2.idct(spectrum) + 128 # for result checking

# calculate the DCT transformation matrix to speed up
def calTransformMat(N):
	x = np.arange(N).reshape((N, 1))
	y = (2 * x + 1).reshape((1, N))
	A = np.cos(pi * x * y / float(2 * N))
	A[0,:] = [sqrt(1 / float(2))] * N
	A = sqrt(2 / float(N)) * A
	return np.mat(A)

# test DCT and IDCT using example from class
def test():
	testData = []
	for line in open(r".\testdata\example.txt"):
		testData.append([int(x)for x in line.strip().split()])
	testData = np.mat(testData, dtype="double")

	dctMat = calTransformMat(testData.shape[0])
	dctData = DCT(testData, dctMat)
	print dctData
	print IDCT(dctData, dctMat)

def readInTable(datafile):
	quantiTable = []
	for line in open(quantiTableFile):
		quantiTable.append([int(x) for x in line.strip().split()])
	return np.mat(quantiTable)

def quantization(blocks, table):
	resultBlocks = []
	for block in blocks:
		resultBlocks.append(block / quantiTable)
	roundBlocks(resultBlocks)
	return resultBlocks

def inv_quantization(blocks, table):
	# inverse quantization
	resultBlocks = []
	for block in blocks:
		resultBlocks.append(np.array(block) * np.array(table))
	return resultBlocks

def dct(blocks, dctMat, dctMat_inv):
	resultBlocks = []
	for block in blocks:
		resultBlocks.append(DCT(block, dctMat, dctMat_inv))
	return resultBlocks

def idct(blocks, dctMat, dctMat_inv):
	resultBlocks = []
	for block in blocks:
		resultBlocks.append(IDCT(block, dctMat, dctMat_inv))
	return resultBlocks

def roundMat(A):
	row, col = A.shape
	for i in xrange(row):
		for j in xrange(col):
			A[i, j] = round(A[i, j])

def roundBlocks(blocks):
	for block in blocks:
		roundMat(block)

def writeBlock(block, fout):
	row, col = block.shape
	for i in xrange(row):
		for j in xrange(col):
			fout.write("%6d " % block[i,j])
		fout.write("\n")

def writeFloatBlock(block, fout):
	row, col = block.shape
	for i in xrange(row):
		for j in xrange(col):
			fout.write("%6.1f " % block[i,j])
		fout.write("\n")

def saveResult(originBlocks, dctBlocks, quantiTable, quantiBlocks, invQuantiBlocks, idctBlocks):
	numBlocks = len(originBlocks)
	for i in xrange(numBlocks):
		fout = open("result%d.txt" % i, "w")
		fout.write("original block:\n")
		writeBlock(originBlocks[i], fout)
		fout.write("\nDCT result:\n")
		writeFloatBlock(dctBlocks[i], fout)
		fout.write("\nQuantization table:\n")
		writeFloatBlock(quantiTable, fout)
		fout.write("\nQuantization result:\n")
		writeFloatBlock(quantiBlocks[i], fout)
		fout.write("\nInvQuantization result:\n")
		writeFloatBlock(invQuantiBlocks[i], fout)
		fout.write("\nIDCT result:\n")
		writeBlock(idctBlocks[i], fout)
	fout.close()

def combineBlocks(blocks, width, height, blocksize):
	numColBlocks = width / blocksize + min(width % blocksize, 1)
	numRowBlocks = len(blocks) / numColBlocks
	img = np.zeros((numRowBlocks * blocksize, numColBlocks * blocksize), dtype = "uint8")
	ctr = 0
	for i in range(0, height, blocksize):
		for j in xrange(0, width, blocksize):
			img[i:i+blocksize, j:j+blocksize] = blocks[ctr][:,:]
			ctr += 1
	return img[:height, :width]


if __name__ == '__main__':
	#test()

	datafile = "data.txt"
	quantiTableFile = "quantizationTable.txt"
	width = height = 16
	blocksize = 8

	# read data in
	pixels = readData(datafile)
	img = arrayToMat(pixels, width, height)
	#cv2.imwrite("input.jpg", img)

	# divided into blocks
	originBlocks = divideIntoBlock(img, blocksize)
	#print originBlocks[0]
	# DCT
	dctMat = calTransformMat(blocksize)
	dctMat_inv = dctMat.I
	dctBlocks = dct(originBlocks, dctMat, dctMat_inv)
	print dctBlocks[0]
	
	# quantization
	quantiTable = readInTable(quantiTableFile)
	quantiBlocks = quantization(dctBlocks, quantiTable)
	print quantiBlocks[0]
	# inverse quantization
	invQuantiBlocks = inv_quantization(quantiBlocks, quantiTable)
	print invQuantiBlocks[0]

	# IDCT
	idctBlocks = idct(invQuantiBlocks, dctMat, dctMat_inv)
	# rounding
	roundBlocks(idctBlocks)
	cv2.imwrite("output.jpg", combineBlocks(idctBlocks, width, height, blocksize))
	print idctBlocks[0]
	#sys.exit()
	saveResult(originBlocks, dctBlocks, quantiTable, quantiBlocks, invQuantiBlocks, idctBlocks)