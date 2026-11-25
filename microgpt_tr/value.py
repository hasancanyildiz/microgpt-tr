"""Skaler değerler üzerinde otomatik türev hesabı yapar.
İşlemlerin sonuçlarını ve yerel türevlerini saklayarak geriye yayılım sağlar.
"""

import math


class Value:
    """Bir sayıyı ve hesaplama grafiğindeki gradyanını tutar."""

    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data + other.data,
            (self, other),
            (1.0, 1.0),
        )

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data * other.data,
            (self, other),
            (other.data, self.data),
        )

    def __pow__(self, power):
        return Value(
            self.data ** power,
            (self,),
            (power * self.data ** (power - 1),),
        )

    def log(self):
        return Value(
            math.log(self.data),
            (self,),
            (1.0 / self.data,),
        )

    def exp(self):
        result = math.exp(self.data)
        return Value(
            result,
            (self,),
            (result,),
        )

    def relu(self):
        return Value(
            max(0.0, self.data),
            (self,),
            (float(self.data > 0),),
        )

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        """Hesaplama grafiğini geriye dolaşıp gradyanları biriktirir."""
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            for child, local_grad in zip(
                node._children,
                node._local_grads,
            ):
                child.grad += local_grad * node.grad
