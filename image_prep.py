def crop_img(img, coord1, coord2):
    cropped = img[coord1[1]:coord2[1], coord1[0]:coord2[0]]
    return cropped.copy()
