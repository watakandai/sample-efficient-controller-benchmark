import numpy as np
from typing import List, Type
from abc import ABCMeta, abstractmethod
from .base import Controller


class VoronoiNode:
    """Node of the Voronoi Treemap
    A node contains the current datapoint in the region.
    """
    x: List[float]
    u: List[float]
    A: np.ndarray
    b: np.array
    alpha: np.ndarray
    beta: float
    parent: Type['VoronoiNode']
    left: Type['VoronoiNode']
    right: Type['VoronoiNode']

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
    def computeHyperPlane(x1, x2):
        pass

    def isLeft(self, x) -> bool:
        return np.dot(self.alpha, x) <= self.beta

    def isLeaf(self) -> bool:
        if self.left is None and self.right is None:
            return True
        return False


class VoronoiController(Controller):
    """Controller struct"""
    root: VoronoiNode
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
        self.root.addPoint(x, u)
