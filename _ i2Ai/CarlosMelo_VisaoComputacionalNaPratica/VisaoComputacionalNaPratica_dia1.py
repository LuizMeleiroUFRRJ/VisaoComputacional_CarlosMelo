import cv2

# Carregando imagem RGB e segmentando canais
imagem = cv2.imread("Fig4_1.png")
azul, verde, vermelho = cv2.split(imagem)

# Exibindo imagens dos canais separados
cv2.imshow("Canal R", vermelho)
cv2.imshow("Canal G", verde)
cv2.imshow("Canal B", azul)
cv2.imshow("Figura 4.1", imagem)

cv2.waitKey(0)
cv2.destroyAllWindows()
