import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma
import skimage.io
# import sys
import matplotlib.pyplot as plt
import cv2
from uuid import uuid4

import config



def mostrarImagem(titulo, imagem):
    """
    Abstrair funções de mostrar a imagem e excluir a tela.
    :param titulo string
    :param imagem imagem
    """
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.imshow(titulo, imagem)
    cv2.waitKey(0)
    cv2.destroyWindow(titulo)


def label2gray(labels):
  ##"""
  ##Convert a labels image to an rgb image using a matplotlib colormap
  ##"""
  label_range = np.linspace(0, 1, 256)
  lut = np.uint8(plt.cm.gray(label_range)[:,2::-1]*256).reshape(256, 1, 3) # replace viridis with a matplotlib colormap of your choice
  return cv2.LUT(cv2.merge((labels, labels, labels)), lut)


def ler_imagem(nomeImagem: str) -> np.ndarray:
    # lê imagem e remove os labels, em escala de cinza
    return skimage.img_as_float(skimage.io.imread(nomeImagem))

def denoiseImagem(imagem: np.ndarray) -> np.ndarray:
    if not config.wavelet:
        return imagem
    sigma_est = estimate_sigma(imagem, average_sigmas=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=3,wavelet='bior6.8', rescale_sigma=True)
    #return denoise_wavelet(imagem,method='VisuShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=5,wavelet='bior6.8', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=6,wavelet='bior6.8', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=6,wavelet='bior6.8', rescale_sigma=True)
    #return denoise_wavelet(imagem,method='VisuShrink', mode='soft', sigma=sigma_est/6, wavelet_levels=5,wavelet='bior6.8', rescale_sigma=True)
    #return denoise_wavelet(imagem,method='VisuShrink', mode='hard', sigma=sigma_est/3, wavelet_levels=5,wavelet='bior6.8', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=8,wavelet='bior6.8', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=3,wavelet='bior1.5', rescale_sigma=True)
    #return denoise_wavelet(imagem, wavelet_levels=3, wavelet='bior1.5', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='bior1.5', rescale_sigma=True) #??
    #return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=3,wavelet='bior1.5', rescale_sigma=True) #??
    #return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=16,wavelet='bior1.5', rescale_sigma=True) #??
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=16,wavelet='bior1.5', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=8,wavelet='bior1.5', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=8,wavelet='bior1.5', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=3,wavelet='haar', rescale_sigma=True)
    #?return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=3,wavelet='haar', rescale_sigma=True)
    #?return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=6,wavelet='haar', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=6,wavelet='haar', rescale_sigma=True)
    #?return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=8,wavelet='haar', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=8,wavelet='haar', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=16,wavelet='haar', rescale_sigma=True)
    #return denoise_wavelet(imagem,method='VisuShrink', mode='hard', sigma=sigma_est/3, wavelet_levels=16,wavelet='haar', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=3,wavelet='sym9', rescale_sigma=True)
    #--return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=3,wavelet='sym9', rescale_sigma=True)
    #--return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=6,wavelet='sym9', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=6,wavelet='sym9', rescale_sigma=True)
    #--return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=8,wavelet='sym9', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=8,wavelet='sym9', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=3,wavelet='Coif3', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=3,wavelet='Coif3', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=6,wavelet='Coif3', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='soft',wavelet_levels=8,wavelet='Coif3', rescale_sigma=True)
    #return denoise_wavelet(imagem,method='VisuShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=5,wavelet='Coif3', rescale_sigma=True)
    #return denoise_wavelet(imagem,method='VisuShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=5,wavelet='sym2', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=3,wavelet='sym2', rescale_sigma=True)
    #return denoise_wavelet(imagem, method='BayesShrink', mode='hard',wavelet_levels=8,wavelet='sym2', rescale_sigma=True)
    #return denoise_wavelet(imagem,method='VisuShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=3,wavelet='Coif3', rescale_sigma=True)
    return denoise_wavelet(imagem,method='VisuShrink', mode='hard', sigma=sigma_est/3, wavelet_levels=5,wavelet='Coif3', rescale_sigma=True)
    

    

def limiariza(imagem: np.ndarray, limiar: float) -> np.ndarray:
    limiarizada = imagem.copy()
    # for i in range(len(limiarizada)):
    #     for j in range(len(limiarizada[i])):
    #         limiarizada[i, j] = 0 if limiarizada[i, j] <= limiar else 255 # TODO: Ver se é isso mesmo
    return limiarizada

def remove_fundo(imagem: np.ndarray, filtro: np.ndarray)-> np.ndarray:
    if not config.fundo:
        return imagem
    semFundo = imagem.copy()
    for i in range(len(semFundo)):
        for j in range(len(semFundo[i])):
            semFundo[i, j] = 0 if filtro[i, j] == 0 else imagem[i, j]
    return semFundo

def salva_imagem(imagem: str) -> str:
    denoise = denoiseImagem(ler_imagem(imagem))
    semFundo = remove_fundo(denoise, limiariza(denoise, 57/255))
    cv2.imwrite(imagem, semFundo)

#if __name__ == '__main__':
def procesamento(imagem: np.ndarray) -> np.ndarray: 
    img = imagem

    semFundo = remove_fundo(img, limiariza(img, 25))
    denoise = denoiseImagem( semFundo )

    if config.salvar:
        plt.figure(figsize=(30,30))
        plt.subplot(2,2,1)
        plt.imshow(img[:,:,0], cmap=plt.cm.gray)
        plt.title('original Image', fontsize =30)
        plt.subplot(2,2,2)
        plt.imshow(denoise[:,:,0], cmap=plt.cm.gray)
        plt.title('Imagem filtrada', fontsize =30)
        plt.savefig(f'/home/eduarda/tcc/figuras_processadas/{uuid4()}.png')
    return(denoise)
