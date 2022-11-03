# -*- coding: utf-8 -*-
'''Fichier de test pour roi_selector.py'''
from roi_selector import *
from PIL import Image

def test_add_pixels(img_path):
    img = Image.open(img_path).convert("RGB")
    img.show()
    original_img_array = np.array(np.array(img))
    print("Old size : ", original_img_array.shape)
    img = img.resize((round(original_img_array.shape[1]/5), round(original_img_array.shape[0]/5))) # resize((width, height))
    img.show()
    resized_img_array = np.array(np.array(img))
    w_edge_img_array = add_pixels(resized_img_array, original_img_array.shape[0], original_img_array.shape[1])
    print("\n New size", w_edge_img_array.shape)
    img = Image.fromarray(w_edge_img_array)
    img.show() 

def test_change_img_edges(img_path):
    # Zoom back test
    old_img = Image.open(img_path).convert("RGB")
    old_img_array = np.array(np.array(old_img))
    zoom_shape = (round(old_img_array.shape[0]/1.5), round(old_img_array.shape[1]/1.5))
    zoom_img = old_img.resize(zoom_shape)
    zoom_img.show()
    zoom_img_array = np.array(zoom_img)
    print("Old size : ", (old_img_array.shape[0], old_img_array.shape[1]))
    new_img = change_img_edges(zoom_img_array, (old_img_array.shape[0], old_img_array.shape[1]), False)
    print("\nNew size",np.array(new_img).shape, "\n")
    new_img.show() 

    # Zoom forward test
    zoom_shape = (round(old_img_array.shape[0]*1.5), round(old_img_array.shape[1]*1.5))
    zoom_img = old_img.resize(zoom_shape)
    zoom_img.show()
    zoom_img_array = np.array(zoom_img)
    print("Old size : ", (old_img_array.shape[0], old_img_array.shape[1]))
    new_img = change_img_edges(zoom_img_array, (old_img_array.shape[0], old_img_array.shape[1]), True)
    print("\nNew size",np.array(new_img).shape, "\n")
    new_img.show() 


def change_img_edges(img_array, old_img_shape, crop):
    '''Crop the image if it has been zoomed. Add a white rectangle if the image has been dezoomed.  
    Parameters
    ----------
    img_array: np array   
        Image that needs to be adjusted
    old_img_shape: tuple 
        (height, width) of the image before the zoom
    crop: bool
       True if the zoom was positive, False if it was negative

    Returns     
    ----------
    img  
        PIL image ready to be displayed after the zoom. '''


    if crop==True:
        # calculate left, top, right and bottom necessary to crop 
        left = (img_array.shape[1] - old_img_shape[1])/2
        top = (img_array.shape[0] - old_img_shape[0])/2
        right = left + old_img_shape[1]
        bottom = top + old_img_shape[0]
        img = Image.fromarray(img_array)
        img = img.crop((left, top, right, bottom)) # crop with Image function 
    else:
        img_array = add_pixels(img_array, old_img_shape[0], old_img_shape[1]) # Add white rectangle to get an image array to shape wanted
        img = Image.fromarray(img_array)


    return img 

def add_pixels(img_array, new_height, new_width):
    '''Add white rectangle to get an image array to shape wanted
    Parameters:
    ----------
    img_array: np.ndarray
        image to which we add the white rectangle
    new_height: int
        height wanted for the new image (= the one of the old image)
    new_width: int
        width wanted for the new image (= the one of the old image)

    returns 
    ----------
    img_array
        Image shaped as wanted with a white rectangle on the edges
    '''
    # ajouter les pixels sur les 3D

    (height, width, channels) = np.shape(img_array)
    high_add = True # on commence par ajouter des pixels par le haut et à gauche et ensuite on alterne avec par le bas et à droite
    while width < new_width or height < new_height:
        if width < new_width and height < new_height: # Si ni la largeur ni la hauteur n'est bonne
            # ajouter selon les 2 axes
            if high_add == True:
                img_array = np.insert(img_array, 0, 255,  axis=0) # cut off line 0
                img_array = np.insert(img_array, 0, 255, axis=1) # cut off column 0
                high_add = False
            else:
                img_array = np.insert(img_array, height-1, 255, axis=0) # cut off last line
                img_array = np.insert(img_array, width-1, 255, axis=1) # cut off last column 
                high_add = True      
        elif width < new_width and height >= new_height: # Si la largeur n'est pas bonne
            # couper selon axis = 1 uniquement
            if high_add == True:
                img_array = np.insert(img_array, 0, 255, axis=1) # cut off 1st column
                high_add = False
            else: 
                img_array = np.insert(img_array, width-1, 255, axis=1) # cut off last column
                high_add = True
        elif width >= new_width and height < new_height: # Si la hauteur n'est pas bonne
            # couper selon axis = 0 uniquement 
            if high_add == True:
                img_array = np.insert(img_array, 0, 255, axis=0) # cut off 1st line
                high_add = False
            else: 
                img_array = np.insert(img_array, height-1, 255, axis=0) # cut off last line
                high_add = True
        (height, width, channels) = np.shape(img_array)
        
    return img_array

def main():
    test_add_pixels("Dataset/training/b4.jpg")
    #test_change_img_edges("Dataset/croped_training/b4.png")

if __name__ == '__main__':
    main()