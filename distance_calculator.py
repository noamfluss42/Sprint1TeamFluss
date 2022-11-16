import math

def interpulate(x):
    pixels_amount_arr = [326, 324, 320, 318, 316, 314, 312, 310, 308, 308, 306, 306, 304, 302, 300, 300, 300, 298, 298, 296, 296, 296, 294]
    distance_arr = [4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 10.25]
    #if x < 296:
    return (0.45*(x**2)-296*x+49210)/100
    #for i in range(len(pixels_amount_arr)):
    #    if pixels_amount_arr[i] <= x:
    #        return (distance_arr[i] + distance_arr[i + 1]) / 2


def distance_in_meters(coord):
    dist = interpulate(coord[0])
    return dist
