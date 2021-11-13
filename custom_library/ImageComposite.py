import numpy as np
import cv2
import matplotlib.pyplot as plt
import PerspectiveLayer as PL

IMAGE_SAVE = True

filecount = [376, 369, 422, 643, 679, 391]

for filenumber in range(724,3000,1):
    folder = np.random.randint(4,10)
    filename = np.random.randint(0,filecount[folder-4])
    bg = cv2.imread('background_hand/%03d/%05d.png'%(folder,filename))
    while True:
        card_path = 'card/new/%04d.png'%(np.random.randint(1,781))
        card = cv2.imread(card_path,cv2.IMREAD_UNCHANGED)
        if card.shape[2] == 4:
            break

    # center crop the bg image
    height, width = bg.shape[:2]
    bg = bg[height//4:3*height//4,:]

    # resize the bg image to 512*512
    bg = cv2.resize(bg,(512,512))
    bg_lb = np.zeros((512,512))
    
    # generate random size, xcor, ycor offset factor
    rd_size = np.random.randint(256,512)
    rd_x = np.random.randint(0,512-rd_size)
    rd_y = np.random.randint(0,512-rd_size)

    # generate warped card image, label of warped card image
    card, card_lb = PL.WarpedPerspectiveImage(card,rd_size)
    # pull out the alpha channel for manual alpha blending
    height, width = card.shape[:2]
    alpha = np.reshape(card[:,:,3]/255,(height,width,1))
    alpha = np.repeat(alpha,3,axis=2)
    card = cv2.cvtColor(card,cv2.COLOR_RGBA2BGR)

    # image brightness by mean. apply normalization
    card = card.astype(np.float64) * (np.mean(bg)/np.mean(card)) * 0.8
    card = np.clip(card,0,255).astype(np.uint8)

    # blurred background image
    bg_blur = cv2.GaussianBlur(bg,(99,99),50)
    bg_blur_alpha = np.ones_like(bg_blur,dtype=np.float64) * 0.15

    # overlap and blending
    bg[rd_y:rd_y+rd_size,rd_x:rd_x+rd_size] = \
        bg[rd_y:rd_y+rd_size,rd_x:rd_x+rd_size]*(1-alpha)+card*alpha
    bg_lb[rd_y:rd_y+rd_size,rd_x:rd_x+rd_size] += card_lb

    # label image type change to greyscale
    bg_lb = (bg_lb*255).astype(np.uint8)

    # gausian blur lighting
    bg = bg*(1-bg_blur_alpha) + bg_blur*bg_blur_alpha
    bg = bg.astype(np.uint8)

    # add noise
    noise = np.random.rand(bg.shape[0]*bg.shape[1]*bg.shape[2],1)
    noise = (noise*8-4).reshape(bg.shape)
    bg_after = np.clip(bg + noise,0,255).astype(np.uint8)

    # add ambient light?
    bg_gauss = cv2.GaussianBlur(bg_after,(3,3),0.5)

    # cv2.imshow('title1',bg_lb)

    # # cv2.imshow('light',bg)
    # # cv2.imshow('noise',bg_after)
    # cv2.imshow('gaussed',bg_gauss)
    # cv2.waitKey()

    if IMAGE_SAVE:
        cv2.imwrite('card_autoencoder_dataset/inputs/%05d.png'%(filenumber),bg_gauss)
        cv2.imwrite('card_autoencoder_dataset/labels/%05d.png'%(filenumber),bg_lb)
        print('%05d.png saved'%(filenumber))