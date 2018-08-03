#coding:utf-8
import numpy as np

def convertAbsScale(src):
    dst = np.clip(src,0,255).astype(np.uint8)
    return dst

def rgb2gray_avg(src):
    assert len(src.shape) == 3
    rgb = np.float32(src)
    r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    dst = (r + g + b) / 3.0
    dst = np.clip(dst,0,255)
    dst = np.uint8(dst)
    return dst

def rgb2gray(src):
    assert len(src.shape) == 3
    rgb = np.float32(src)
    dst = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    dst = np.clip(dst,0,255)
    dst = dst.astype(np.uint8)
    return dst

def rgb2xyz(src):
    assert len(src.shape) == 3
    rgb = np.float32(src) / 255.0
    r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]

    xyz = np.zeros_like(rgb)

    var_R = np.zeros_like(r)
    var_G = np.zeros_like(r)
    var_B = np.zeros_like(r)
    
    idx = r > 0.04045
    var_R[idx] = ((r[idx] + 0.055) / 1.055) ** 2
    var_R[~idx] = r[~idx] / 12.92
    idx = g > 0.04045
    var_G[idx] = ((g[idx] + 0.055) / 1.055) ** 2
    var_G[~idx] = g[~idx] / 12.92
    idx = b > 0.04045
    var_B[idx] = ((b[idx] + 0.055) / 1.055) ** 2
    var_B[~idx] = b[~idx] / 12.92

    var_R = var_R * 100.0
    var_G = var_G * 100.0
    var_B = var_B * 100.0

    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    xyz[:,:,0] = X
    xyz[:,:,1] = Y
    xyz[:,:,2] = Z

    # xyz = np.clip(xyz,0,255)
    # xyz = np.uint8(xyz)
    return xyz

def xyz2rgb(src):
    assert len(src.shape) == 3
    xyz = np.float64(src) / 100.0
    rgb = np.zeros_like(xyz)
    X,Y,Z = xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]
    var_R = X * 3.2406 - Y * 1.5372 - Z * 0.4986
    var_G = X * (-0.9689) + Y * 1.87597 + Z * 0.03342
    var_B = X * 0.01344 - Y * 0.11836 + Z * 1.34926
    #eps = 1.0e-5
    pd =  1.0 / 2.19921875 
    var_R = var_R ** pd #( 1.0 / 2.19921875 )
    var_G = var_G ** pd #( 1.0 / 2.19921875 )
    var_B = var_B ** pd #( 1.0 / 2.19921875 )

    rgb[:,:,0] = var_R * 255.0
    rgb[:,:,1] = var_G * 255.0
    rgb[:,:,2] = var_B * 255.0

    # rgb = np.clip(rgb,0,255)
    # rgb = np.uint8(rgb)
    return rgb
# Use illuminants and observer D50(CIE 1964)
# X1 = 96.720 Y1 = 100.000 Z1 = 81.427
# X2 = 96.422	 Y2 = 100.000 Z2 = 82.521
def xyz2lab(src):
    assert len(src.shape) == 3
    xyz = np.float32(src)
    lab = np.zeros_like(xyz)
    ref_x ,ref_y,ref_z = 96.720, 100.000, 81.427
    x,y,z = xyz[:,:,0] / ref_x,xyz[:,:,1] / ref_y,xyz[:,:,2] / ref_z
    var_x = np.zeros_like(x)
    var_y = np.zeros_like(x)
    var_z = np.zeros_like(x)

    idx = x > 0.008856
    var_x[idx] = x[idx] ** (1 / 3.0)
    var_x[~idx] = x[~idx] * 7.787 + 16.0 / 116.0

    idx = y > 0.008856
    var_y[idx] = y[idx] ** (1 / 3.0)
    var_y[~idx] = y[~idx] * 7.787 + 16.0 / 116.0

    idx = z > 0.008856
    var_z[idx] = z[idx] ** (1 / 3.0)
    var_z[~idx] = z[~idx] * 7.787 + 16.0 / 116.0

    lab[:,:,0] = (116 * var_x) - 16.0
    lab[:,:,1] = 500 * (var_x - var_y)
    lab[:,:,2] = 200 * (var_y - var_z)

    # lab = np.clip(lab,0,255)
    # lab = np.uint8(lab)
    return lab

def lab2xyz(src):
    assert len(src.shape) == 3
    lab = np.float32(src)
    L,A,B = lab[:,:,0],lab[:,:,1],lab[:,:,2]
    var_y = (L + 16.0) / 116.0
    var_x = A / 500.0 + var_y
    var_z = var_y - B / 200.0

    xyz = np.zeros_like(lab)
    X = np.zeros_like(L)
    Y = np.zeros_like(L)
    Z = np.zeros_like(L)
    vy = var_y ** 3
    idx = (vy > 0.008856)
    Y[idx] = var_y[idx] ** 3
    Y[~idx] = (var_y[~idx] - 16 / 116) / 7.787

    vx = var_x ** 3
    idx = (vx > 0.008856)
    X[idx] = var_x[idx] ** 3
    X[~idx] = (var_x[~idx] - 16 / 116) / 7.787

    vz = var_z ** 3
    idx = (vz > 0.008856)
    Z[idx] = var_z[idx] ** 3
    Z[~idx] = (var_z[~idx] - 16 / 116) / 7.787
    
    ref_x ,ref_y,ref_z = 96.720, 100.000, 81.427
   

    X = X * ref_x
    Y = Y * ref_y
    Z = Z * ref_z

    xyz[:,:,0] = X
    xyz[:,:,1] = Y
    xyz[:,:,2] = Z

    # xyz = np.clip(xyz,0,255)
    # xyz = np.uint8(xyz)
    return xyz

def rgb2lab(src):
    return xyz2lab(rgb2xyz(src))

def lab2rgb(src):
    return xyz2rgb(lab2xyz(src))