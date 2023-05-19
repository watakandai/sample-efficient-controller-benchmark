import os
import time
import numpy as np
from pathlib import Path
from typing import List, Type
from abc import ABCMeta, abstractmethod
from .base import Controller


class VoronoiNode:
    """Node of the Voronoi Treemap
    A node contains the current datapoint in the region.
    """
    x: List[float] = None
    u: List[float] = None
    A: np.ndarray = None
    b: np.array = None
    alpha: np.ndarray = None
    beta: float = 0.0
    parent: Type['VoronoiNode'] = None
    left: Type['VoronoiNode'] = None
    right: Type['VoronoiNode'] = None

    def __init__(self, x, u, A, b):
        self.x = x
        self.u = u
        self.A = A
        self.b = b

    def addPoint(self, x, u) -> None:
        if self.isLeaf():
            # compute a separating plane a^T * x <= b
            self.alpha, self.beta = self.computeHyperPlane(self.x, x)

            # Left is the original point with the additional hyperplane
            A = np.r_[self.A, self.alpha]
            b = np.r_[self.b, self.beta]
            self.left = VoronoiNode(self.x, u, A, b)
            self.left.parent = self

            # Right is the new point with the additional hyperplane
            A = np.r_[self.A, -self.alpha]
            b = np.r_[self.b, -self.beta]
            self.right = VoronoiNode(x, u, A, b)
            self.right.parent = self
        else:
            if self.isLeft(x):
                self.left.addPoint(x, u)
            else:
                self.right.addPoint(x, u)

    # TODO:
    @staticmethod
    def computeHyperPlane(x1: np.ndarray, x2: np.ndarray):
        """Compute the hyperplane between two points
        a^T * x + b = 0, then x is on the hyperplane
        a^T * x + b > 0, if x is on x2's side
        a^T * x + b < 0, if x is on x1's side

        Args:
            x1 (np.ndarray): _description_
            x2 (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        if all(x1 == x2):
            raise Exception("Two points cannot be the same for computing hyperplane")

        # Compute the vector between the two points
        v = x2 - x1
        # Compute the unit vector in the direction of v
        u = v / np.linalg.norm(v)
        # Compute the equation of the hyperplane
        a = u
        xc = (x1 + x2) / 2
        b = -np.dot(u, xc)
        return a, b

    def isLeft(self, x) -> bool:
        return np.dot(self.alpha, x) <= self.beta

    def isLeaf(self) -> bool:
        if self.left is None and self.right is None:
            return True
        return False


class VoronoiTreeController(Controller):
    """Controller struct"""
    root: VoronoiNode = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def action(self, x: List[float]) -> List[float]:

        # Find teh nearest point
        if self.root is None:
            raise Exception("No data point is added")
        node = self._nearest_neighbor(self.root, x)

        # return its control input
        return node.u

    def _nearest_neighbor(self, n: VoronoiNode, x: List[float]) -> VoronoiNode:
        if n.isLeaf():
            return n

        if n.isLeft(x):
            return self._nearest_neighbor(n.left, x)
        else:
            return self._nearest_neighbor(n.right, x)

    def update(self, X, U, Dt) -> List[float]:
        if len(X) == 0 or len(U) == 0:
            raise Exception("A trajectory cannot be empty")

        x, u = X[0], U[0]

        if self.root is None:
            self.root = VoronoiNode(x, u, np.array([]), np.array([]))
        else:
            self.root.addPoint(x, u)


class VoronoiController(Controller):
    """Controller struct"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X = []
        self.nX = []
        self.U = []
        self.Dt = []

    def action(self, x: List[float]) -> List[float]:
        if len(self.X) == 0:
            raise Exception("No data point is added")

        idx = np.argmin(list(map(lambda x_: np.linalg.norm(x_ - x), self.X)))
        return {"control": self.U[idx], "dt": self.Dt[idx]}

    def update(self, X, U, Dt) -> List[float]:
        if len(X) == 0 or len(U) == 0:
            print("Skipping ... The trajectory is empty")
            return

        x, u = X[0], U[0]
        nx = X[1]
        dt = Dt[0]

        self.X.append(x)
        self.U.append(u)
        self.nX.append(nx)
        self.Dt.append(dt)

    def save(self, filedir: str=None):
        Path(filedir).parent.mkdir(parents=True, exist_ok=True)

        fileX = os.path.join(filedir, "X")
        fileU = os.path.join(filedir, "U")
        filenX = os.path.join(filedir, "nX")
        fileDt = os.path.join(filedir, "Dt")

        np.save(fileX, np.array(self.X, dtype=object), allow_pickle=True)
        np.save(fileU, np.array(self.U, dtype=object), allow_pickle=True)
        np.save(filenX, np.array(self.nX, dtype=object), allow_pickle=True)
        np.save(fileDt, np.array(self.Dt, dtype=object), allow_pickle=True)

    def load(self, filedir: str=None):
        fileX = os.path.join(filedir, "X")
        fileU = os.path.join(filedir, "U")
        filenX = os.path.join(filedir, "nX")
        fileDt = os.path.join(filedir, "Dt")

        self.X = np.load(fileX, allow_pickle=True)
        self.U = np.load(fileU, allow_pickle=True)
        self.nX = np.load(filenX, allow_pickle=True)
        self.Dt = np.load(fileDt, allow_pickle=True)
