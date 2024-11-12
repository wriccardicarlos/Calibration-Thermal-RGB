import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do tabuleiro
rows, cols = 6, 9  # Número de quadrados
square_size = 50   # Tamanho do quadrado em pixels

# Cria uma imagem em branco com três canais de cor (BGR)
board = np.zeros((rows * square_size, cols * square_size, 3), dtype=np.uint8)

# Desenha o padrão de xadrez com vermelho e azul
for i in range(rows):
    for j in range(cols):
        color = (0, 0, 255) if (i + j) % 2 == 0 else (255, 0, 0)  # Vermelho e azul
        cv2.rectangle(
            board, 
            (j * square_size, i * square_size), 
            ((j + 1) * square_size, (i + 1) * square_size), 
            color, 
            -1
        )

# Salve a imagem para verificar
cv2.imwrite("chessboardColorido.jpg", board)

# Mostrar a imagem
cv2.imshow("Tabuleiro", board)
cv2.waitKey(0)
cv2.destroyAllWindows()
