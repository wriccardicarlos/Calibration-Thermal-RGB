import cv2
import numpy as np

# Defina o tamanho do tabuleiro (9x6 quadrados visíveis)
rows, cols = 9, 6
square_size = 50  # Tamanho do quadrado em pixels

# Crie uma imagem em branco
board = np.zeros((rows * square_size, cols * square_size), dtype=np.uint8)

# Desenhe o padrão de xadrez
for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            cv2.rectangle(
                board, 
                (j * square_size, i * square_size), 
                ((j + 1) * square_size, (i + 1) * square_size), 
                255, 
                -1
            )

# Salve a imagem para verificar
cv2.imwrite("chessboard.jpg", board)

# Mostrar a imagem
cv2.imshow("Tabuleiro", board)
cv2.waitKey(0)
cv2.destroyAllWindows()
