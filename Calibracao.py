import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def calibrate(showPics=True):
    # Ler a imagem
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'calibration')
    list_of_image_files = glob.glob(os.path.join(calibrationDir,'*.jpg'))
    #list_of_image_files = glob.glob('chessboard.jpg') # usado para pegar a imagem de modo individual (testes)
    if list_of_image_files is None:
        print("No images found in the directory")
        exit()

    # Parâmetros do tabuleiro de xadrez
    nRows, nCols = 6, 4
    termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Preparar pontos do objeto 3D
    objp = np.zeros((nRows*nCols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1, 2)

    # Listas para armazenar pontos do objeto 3D e pontos da imagem 2D
    object_points = []
    image_points = []

    # Carregar e processar cada imagem
    for image_file in list_of_image_files:
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detectar cantos do tabuleiro de xadrez
        ret, corners = cv2.findChessboardCorners(image,(nRows,nCols))

        # Se os cantos forem encontrados, adicione os pontos do objeto e da imagem
        if ret:
            object_points.append(objp)
            cornersRefined = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),termCriteria)
            image_points.append(cornersRefined)

            if showPics:
                # Desenhar e exibir os cantos
                cv2.drawChessboardCorners(image, (nRows,nCols), cornersRefined, ret)
                cv2.imshow('ChessBoard', image)
                cv2.waitKey(500)
        else: 
            print('No chessboard detected for image: ', image_file)
    cv2.destroyAllWindows()

    # Calibrar a câmera
    repError, camMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    print("Matriz de calibração K:\n", camMatrix)
    print("Distorção:", dist.ravel())
    print("Reproj Error (pixels): {:.4f}".format(repError))

    #salva os dados de calibração em um arquivo
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder,'calibracaoUnDist.npz')
    np.savez(paramPath,
             repError=repError,
             camMatrix=camMatrix,
             dist=dist,
             rvecs=rvecs,
             tvecs=tvecs)
    return camMatrix,dist

def removeDistorcion(camMatrix,dist):
    root = os.getcwd()
    imgpath = os.path.join(root,'calibration/testeDist.jpg')
    img = cv2.imread(imgpath)

    heigth, width = img.shape[:2]

    camMatrixNew,roi = cv2.getOptimalNewCameraMatrix(camMatrix,dist,(width,heigth),1,(width,heigth))
    imgUndist = cv2.undistort(img,camMatrix,dist,None,camMatrixNew)

    # escrever linha para ver a distorçao
    cv2.line(img,(1769,103),(1780,922),(255,255,255),2)
    cv2.line(imgUndist,(1769,103),(1780,922),(255,255,255),2)
    cv2.imwrite("imgUndist.jpg", imgUndist)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()

def runCalibration():
    calibrate(showPics=True)

def runRemoveDistortion():
    camMatrix, dist = calibrate(showPics=False)
    removeDistorcion(camMatrix,dist)

if  __name__ == "__main__":
    runCalibration()
    #runRemoveDistortion()
