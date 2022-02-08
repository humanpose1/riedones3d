import torch


def eulerAnglesToRotationMatrix(theta: torch.Tensor):
    """
taken from
    """

    ctx = torch.cos(theta[0])
    stx = torch.sin(theta[0])
    cty = torch.cos(theta[1])
    sty = torch.sin(theta[1])
    ctz = torch.cos(theta[2])
    stz = torch.sin(theta[2])
    o = torch.zeros_like(ctx)
    one = torch.ones_like(ctx)
    R_x = torch.tensor([[one, o, o],
                        [o, ctx, -stx],
                        [o, stx, ctx]])

    R_y = torch.tensor([[cty, o, sty],
                        [o, one, o],
                        [-sty, 0, cty]])

    R_z = torch.tensor([[ctz, -stz, o],
                        [stz, ctz, o],
                        [o, o, one]])
    R = R_z @  R_y  @ R_x
    return R


def get_cross_product_matrix(k: torch.Tensor):
    o = torch.zeros_like(k[0])
    return torch.tensor([[o, -k[2], k[1]],
                         [k[2], o, -k[0]],
                         [-k[1], k[0], o]])


def get_vector_from_cross_product_matrix(A: torch.Tensor):
    return torch.tensor([A[2, 1], A[0, 2], A[1, 0]])


def rodrigues(k: torch.Tensor, theta: float):
    """
    compute the rotation matrix using an axis and angle
    source : https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    K = get_cross_product_matrix(k)
    K2 = K @ K
    R = torch.eye(3) + torch.sin(torch.tensor(theta)) * K + (1 - torch.cos(torch.tensor(theta))) * K2
    T = torch.eye(4)
    T[:3, :3] = R
    return T


def compute_T(r: torch.Tensor, t: torch.Tensor, is_rodrigues: bool = True):
    T = torch.eye(4)
    if(len(r) == 1):
        ax = torch.tensor([0, 0, 1]).to(r)
        T = rodrigues(ax, r)
    elif(len(r) == 3):
        if(torch.norm(r) > 0):
            if(is_rodrigues):
                T = rodrigues(r/torch.norm(r), torch.norm(r))
            else:
                T[:3, :3] = eulerAnglesToRotationMatrix(r)

    T[:len(t), 3] = t
    return T


def log_SO3(R: torch.Tensor):
    """
    see here for more details : https://en.wikipedia.org/wiki/3D_rotation_group
    https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    """
    z = 0.5*(R-R.T)
    theta = torch.acos((torch.trace(R)-1)/2)
    if theta > 1e-3:
        log = theta / (torch.sin(theta)) * z
    else:
        return torch.zeros(3)
    return get_vector_from_cross_product_matrix(log)

def vec_2_transfo(vec1, vec2) -> torch.Tensor:
    """
    find the rotation, to go from vec1 to vec2
    """
    assert len(vec1) == 3
    assert len(vec2) == 3
    assert torch.allclose(torch.norm(vec1, dim=0), torch.tensor(1.0))
    assert torch.allclose(torch.norm(vec2, dim=0), torch.tensor(1.0))
    theta = torch.acos(
        vec1.dot(vec2)/(torch.norm(vec1)*torch.norm(vec2)))
    k = torch.cross(vec1, vec2)
    k = k / torch.norm(k)
    T = rodrigues(k, theta.item())
    return T

def batch_vec_2_transfo(V1: torch.Tensor, V2: torch.Tensor) -> torch.Tensor:
    """
    find the rotation for element for V1 and V2
    """
    pass

def compute_PCA(points: torch.Tensor):
    """
    compute the eigenvalues and eigenvectors
    points: torch.Tensor of size N x 3
    """
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    assert points.shape[0] > 0
    mean = points.mean(0)

    covariance_matrix =  ((points - mean).T @ (points - mean)) / points.shape[0]  # size 3 x 3

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
    _, ind = torch.sort(eigenvalues, descending=True)
    return eigenvalues[ind], eigenvectors[:, ind]
