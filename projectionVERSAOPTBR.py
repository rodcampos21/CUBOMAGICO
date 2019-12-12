import numpy as np

class Quaternion:
    """Rotacao quaternioes :

    Classe para ajudar a representar rotacoes 3d via quaternios
    """
    @classmethod
    def from_v_theta(cls, v, theta):
        """
		
        Constroi  quaternios de unidades de vetores v  e rotacoes de angulos theta

        Parametros
        ----------
        v : array_like
            array de vetores, ultima dimensao 3. vetores serao normalizados.
        theta : array_like
            array de rotacoes de angulos em radianos, shape = v.shape[:-1].

        Retorna
        -------
        q : objeto de quaternioes
            quaternioes  representando rotacoes 
        """
        theta = np.asarray(theta)
        v = np.asarray(v)
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)

        v = v * s / np.sqrt(np.sum(v * v, -1))
        x_shape = v.shape[:-1] + (4,)

        x = np.ones(x_shape).reshape(-1, 4)
        x[:, 0] = c.ravel()
        x[:, 1:] = v.reshape(-1, 3)
        x = x.reshape(x_shape)

        return cls(x)

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    def __repr__(self):
        return "Quaternioes:\n" + self.x.__repr__()

    def __mul__(self, other):
        # multiplicacao de dois quaternios
        # nao implementamos multiplicacao escalar
        sxr = self.x.reshape(self.x.shape[:-1] + (4, 1))
        oxr = other.x.reshape(other.x.shape[:-1] + (1, 4))

        prod = sxr * oxr
        return_shape = prod.shape[:-1]
        prod = prod.reshape((-1, 4, 4)).transpose((1, 2, 0))

        ret = np.array([(prod[0, 0] - prod[1, 1]
                         - prod[2, 2] - prod[3, 3]),
                        (prod[0, 1] + prod[1, 0]
                         + prod[2, 3] - prod[3, 2]),
                        (prod[0, 2] - prod[1, 3]
                         + prod[2, 0] + prod[3, 1]),
                        (prod[0, 3] + prod[1, 2]
                         - prod[2, 1] + prod[3, 0])],
                       dtype=np.float,
                       order='F').T
        return self.__class__(ret.reshape(return_shape))

    def as_v_theta(self):
        """Retorna v , theta equivalente de umm Quaternio normalizado"""
        x = self.x.reshape((-1, 4)).T

        # compute theta
        norm = np.sqrt((x ** 2).sum(0))
        theta = 2 * np.arccos(x[0] / norm)

        # compute the unit vector
        v = np.array(x[1:], order='F', copy=True)
        v /= np.sqrt(np.sum(v ** 2, 0))

        # reshape the results
        v = v.T.reshape(self.x.shape[:-1] + (3,))
        theta = theta.reshape(self.x.shape[:-1])

        return v, theta

    def as_rotation_matrix(self):
        """Retorna a rotacao da matriz de um quaternio normalizado"""
        v, theta = self.as_v_theta()

        shape = theta.shape
        theta = theta.reshape(-1)
        v = v.reshape(-1, 3).T
        c = np.cos(theta)
        s = np.sin(theta)

        mat = np.array([[v[0] * v[0] * (1. - c) + c,
                         v[0] * v[1] * (1. - c) - v[2] * s,
                         v[0] * v[2] * (1. - c) + v[1] * s],
                        [v[1] * v[0] * (1. - c) + v[2] * s,
                         v[1] * v[1] * (1. - c) + c,
                         v[1] * v[2] * (1. - c) - v[0] * s],
                        [v[2] * v[0] * (1. - c) - v[1] * s,
                         v[2] * v[1] * (1. - c) + v[0] * s,
                         v[2] * v[2] * (1. - c) + c]],
                       order='F').T
        return mat.reshape(shape + (3, 3))

    def rotate(self, points):
        M = self.as_rotation_matrix()
        return np.dot(points, M.T)


def project_points(points, q, view, vertical=[0, 1, 0]):
    """Projeta pontos usando um quaternio q e uma visao v

    Parametros
    ----------
    pontos : array_like
        array de ultima dimensao 3
    q : Quaternio
        quaternio que representa uma rotacao
    view : array_like
        vetor de tamanho 3 dando um campo de visao 
    vertical : array_like
        direcao de um eixo y para visao. Um erro vira disso
        paralelo a visao

    Retorna
    -------
    proj: array_like
        array de pontos projetados : mesmo formato que os pontos 
    """
    points = np.asarray(points)
    view = np.asarray(view)

    xdir = np.cross(vertical, view).astype(float)

    if np.all(xdir == 0):
        raise ValueError("vertical paralela a v")

    xdir /= np.sqrt(np.dot(xdir, xdir))

    # pega o vetor unitario correspondente a vertical 
    ydir = np.cross(view, xdir)
    ydir /= np.sqrt(np.dot(ydir, ydir))

    # normaliza a vista do local:  o eixo z
    v2 = np.dot(view, view)
    zdir = view / np.sqrt(v2)

    # rotaciona os pontos
    R = q.as_rotation_matrix()
    Rpts = np.dot(points, R.T)

    # projeta os pontos na visao
    dpoint = Rpts - view
    dpoint_view = np.dot(dpoint, view).reshape(dpoint.shape[:-1] + (1,))
    dproj = -dpoint * v2 / dpoint_view

    trans =  list(range(1, dproj.ndim)) + [0]
    return np.array([np.dot(dproj, xdir),
                     np.dot(dproj, ydir),
                     -np.dot(dpoint, zdir)]).transpose(trans)
