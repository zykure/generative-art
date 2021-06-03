#!/usr/bin/env python3

from PIL import Image
from scipy import interpolate
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import sys

# TODO: determine distance from origin (n_rotations)

def random1d(rng, n):
    # create a 2D array of random numbers
    x = np.linspace(0, 1, n)
    z = rng.random(n)

    # return an interpolation function
    f = interpolate.UnivariateSpline(x, z, k=3)

    if DEBUG:
        xr = np.linspace(0, 1, 100)
        zr = np.array([f(xr)])
        plt.imshow(zr, #extent=(yr.min(), yr.max(), xr.max(), xr.min()),
           interpolation='nearest', cmap='gist_rainbow', vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()


    return f

def random2d(rng, n, m):
    # create a 2D array of random numbers
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 2*np.pi, m+1)
    #z = rng.random((m,n))
    z = rng.normal(loc=rng.random(), scale=0.5*rng.random(), size=(m,n))

    # avoid invalid color values
    # based on lmfit, see: https://lmfit.github.io/lmfit-py/bounds.html
    z = (np.sin(z) + 1) * 0.5

    # make it periodic in y
    z = np.vstack([z, z[0,:]])

    # return an interpolation function
    f = interpolate.RectBivariateSpline(x, y, z.T, kx=3, ky=3)

    if DEBUG:
        plt.title('n=%d m=%d seed=%d' % (n,m,seed))
        xx, yy = np.meshgrid(x, y)
        #plt.scatter(xx, yy, c='k', marker='o')
        #plt.scatter(xx, yy, c=z, marker='.', cmap='gist_rainbow')
        xr = np.linspace(0, 1, 100)
        yr = np.linspace(0, 2*np.pi, 100)
        zr = f(xr, yr)
        zr[zr < 0] = np.nan
        zr[zr > 1] = np.nan
        plt.imshow(zr, interpolation='nearest', cmap='gist_rainbow', vmin=0, vmax=1)
        plt.tight_layout()
        plt.show()

    return f

def spiral(x, y, a=1.0, b=1.0, r_max=100, t_start=0):

    # calculate the target radius and theta
    r = np.sqrt(x*x + y*y)
    t = np.arctan2(y, x)
    t += t_start
    if t < 0:
        t += 2*np.pi
    if t >= 2*np.pi:
        t -= 2*np.pi

    # early exit if the point requested is the origin itself
    # to avoid taking the logarithm of zero in the next step
    if r == 0:
        return (0,0,0)

    if r > r_max:
        return (0,0,1)

    # calculate the floating point approximation for n
    n = (np.log(r/a)/b - t) / (2*np.pi)

    # find the two possible radii for the closest point
    upper_r = a * np.power(np.e, b * (t + 2*np.pi*np.ceil(n)))
    lower_r = a * np.power(np.e, b * (t + 2*np.pi*np.floor(n)))
    width = upper_r - lower_r

    if width <= 0:
        return (0,0,0)

    #r_dist = min(abs(upper_r - r), abs(r - lower_r))
    r_dist = abs(r - lower_r)
    r_scale = r_dist / width

    t_scale = t

    if r_scale < 0.02 + 0.01 * random_R(t / (2*np.pi)):
        return (0,0,0)

    # return the minimum distance to the target point
    return (
        random_H(r_scale, t_scale),
        0.2 + 0.6 * random_S(r_scale, 0) + 0.2 * random_S(0, t_scale),
        0.5 + 0.5 * random_V(np.sqrt(r_scale), 0),
    )

# produce a PNM image of the result
if __name__ == '__main__':

    N_ROWS_COLS = 2
    IMG_SIZE = 400
    OUTPUT_SIZE = IMG_SIZE * N_ROWS_COLS

    DEBUG = False
    SEED = np.random.randint(65536)

    OVERSAMPLE = 2
    SIZE = int(OUTPUT_SIZE/2 * OVERSAMPLE) // N_ROWS_COLS
    RATIO = 0.137

    out_im = Image.new('RGB', (OUTPUT_SIZE,OUTPUT_SIZE))

    row_index = 0
    col_index = 0
    for index in range(N_ROWS_COLS**2):

        im = Image.new('HSV', (2*SIZE,2*SIZE))  # try: HSV, LAB, YCbCr

        seed = (SEED * (index+1)) % 65536

        # initialize random generator
        rng = np.random.default_rng(seed)

        random_R = random1d(rng, 11)
        random_H = random2d(rng, 23, 3)
        random_S = random2d(rng, 23, 137)
        random_V = random2d(rng, 11, 5)

        r_max = 0.9 * SIZE
        t_start = rng.random() * 2*np.pi

        n_pixels = (2*SIZE)**2
        print("Generating image #%d with size %dx%d (%dk pixels) [seed=%d] ..." %\
            (index+1, 2*SIZE, 2*SIZE, n_pixels/1e3, seed))

        count = 0
        for x in range(2*SIZE):
            for y in range(2*SIZE):

                i, j = (x-SIZE, y-SIZE)
                hsv = spiral(i, j, a=1.0, b=RATIO, r_max=r_max, t_start=t_start)
                hsv = tuple([ min(255, int(255*x)) for x in hsv ])
                im.putpixel((x,y), hsv)

                count += 1
                if count % 1e3 == 0:
                    print("  Painting ... [%.1f%%]\r" % (100*count/n_pixels), end='', flush=True)
        print("  Painting has finished!")

        out_size = OUTPUT_SIZE // N_ROWS_COLS
        out_name = "output/%d_%dp_%dpx.png" % (SEED, N_ROWS_COLS**2, out_size)
        print("  Resampling image to output size (%dx%d)" % (out_size, out_size))
        im2 = im.convert('RGB').resize((out_size,out_size), resample=Image.HAMMING)

        xp = row_index * out_size
        yp = col_index * out_size
        out_im.paste(im2, (xp,yp))

        row_index += 1
        if (row_index >= N_ROWS_COLS):
            row_index = 0
            col_index += 1

    print("  Saving image to file: %s" % out_name)
    out_im.save(out_name)
    out_im.show()
