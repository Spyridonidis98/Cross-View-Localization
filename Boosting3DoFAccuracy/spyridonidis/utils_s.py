import torch

def render_point_cloud(image_res = 2, camera = 'FL', args = None, net = None, xyz = None):
    """
    returns u:[N], V[N] that can be used with pyplot.scatter 
    """

    height = args.image_H /(2 ** (3 - image_res))
    width = args.image_W/ (2 ** (3 - image_res))

    xyz_w = torch.cat((xyz, torch.ones_like(xyz[-1:])), dim = 0)
    Tcam2carbody = torch.zeros((4,4), dtype=torch.float32)
    Tcam2carbody[:3,:3] = args.Rs[camera]
    Tcam2carbody[:3, 3] = args.Ts[camera]
    Tcam2carbody[3,3] = 1
    Tcarbody2cam = torch.inverse(Tcam2carbody)

    xyz_w_cam= Tcarbody2cam @ xyz_w
    uvs = net.Ks_list[camera][image_res] @ xyz_w_cam[:3, :]
    uvs[0,:] /= uvs[2,:]
    uvs[1,:] /= uvs[2,:]
    bl = (uvs[0,:] >= 0)  & (uvs[0,:] < width-1) & (uvs[1,:] >= 0) & (uvs[1,:] < height-1) & (uvs[2,:] > 0)
    bl = bl.view(-1) * 1
    uvs = uvs[:2, :]
    bl = torch.nonzero(bl)
    uvs = uvs[:, bl].view(2, -1)

    u = uvs[0, :]
    v = uvs[1, :]

    print(f'points rendered for resolution (W = {width},H = {height})')

    return u,v

