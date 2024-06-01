from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
T2 = TypeVar("T2")


class Tree(Iterable[T], ABC):
    """
    Abstract base class for representing trees of a generic value type.
    """

    @abstractmethod
    def map(self, function: Callable[[T], T2]) -> Tree[T2]:
        """
        Applies the ``function`` to the value of each :class:`~torchjd.tree.tree.Leaf` and returns
        the resulting :class:`~torchjd.tree.tree.Tree`.

        :param function: The function to apply to the value of each
            :class:`~torchjd.tree.tree.Leaf`.
        """
        raise NotImplementedError

    @abstractmethod
    def flatmap(self, function: Callable[[T], Tree[T2]]) -> Tree[T2]:
        """
        Applies the ``function`` to the value of each :class:`~torchjd.tree.tree.Leaf` and returns
        the resulting :class:`~torchjd.tree.tree.Tree` flattened, i.e. each leaf having as value a
        tree is replaced by the tree itself.

        :param function: The function to apply to the value of each
            :class:`~torchjd.tree.tree.Leaf`. Its return type must be
            :class:`~torchjd.tree.tree.Tree`.
        """
        raise NotImplementedError

    @abstractmethod
    def filter(self, function: Callable[[T], bool]) -> Tree[T]:
        """
        Filters a Tree, keeping only elements that satisfy ``function``.

        :param function: The function to apply to the value of each
            :class:`~torchjd.tree.tree.Leaf`. Any returned `False` will result in the removal of
            the corresponding element in the returned Tree.
        """
        raise NotImplementedError

    @abstractmethod
    def zip(self, *others: Tree) -> Tree[tuple]:
        """
        Zips self with other trees of the same structure.
        Returns a Tree of the same structure, where leaf values are tuples made of each tree's
        corresponding leaf value.

        :param others: The other trees to zip, their structures have to be the same as `self`.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of leaves contained in the tree.
        """
        raise NotImplementedError


class EmptyTree(Tree[T]):
    """
    Class for representing empty trees of a generic value type.
    """

    def map(self, function: Callable[[T], T2]) -> EmptyTree[T2]:
        return EmptyTree()

    def flatmap(self, function: Callable[[T], Tree[T2]]) -> EmptyTree[T2]:
        return EmptyTree()

    def filter(self, function: Callable[[T], bool]) -> EmptyTree[T]:
        return EmptyTree()

    def zip(self, *others: EmptyTree) -> EmptyTree[tuple]:
        if not all([isinstance(other, EmptyTree) for other in others]):
            raise TypeError("Expected only `EmptyTree`s")

        return EmptyTree()

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[T]:
        return iter(())

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, EmptyTree)

    def __str__(self) -> str:
        return self.__class__.__name__


class NonEmptyTree(Tree[T], ABC):
    """
    Abstract base class for representing non-empty trees of a generic value type.
    """

    @abstractmethod
    def map(self, function: Callable[[T], T2]) -> NonEmptyTree[T2]:
        raise NotImplementedError

    @abstractmethod
    def zip(self, *others: NonEmptyTree) -> NonEmptyTree[tuple]:
        raise NotImplementedError

    @abstractmethod
    def depth(self) -> int:
        """
        Returns the depth of the tree, defined as the length of the longest path to a leaf.
        """
        raise NotImplementedError


class Node(NonEmptyTree[T]):
    """
    Node of a :class:`~torchjd.tree.tree.Tree` that is a parent to at least one other subtree.
    """

    def __init__(self, *children: NonEmptyTree[T]):
        if len(children) == 0:
            raise ValueError("Cannot create a `Node` with no children.")
        self.children = list(children)

    @staticmethod
    def from_trees(*trees: Tree[T]) -> Tree[T]:
        """
        Creates a tree from a collection of trees which are allowed to be empty. This replaces the
        constructor of `Node` when the provided trees can be empty.
        """
        non_empty_children = [child for child in trees if isinstance(child, NonEmptyTree)]
        if len(non_empty_children) == 0:
            return EmptyTree()
        else:
            return Node(*non_empty_children)

    def depth(self) -> int:
        max_value = 0
        for child in self.children:
            depth = child.depth() + 1
            if max_value < depth:
                max_value = depth
        return max_value

    def map(self, function: Callable[[T], T2]) -> Node[T2]:
        return Node(*[child.map(function) for child in self.children])

    def flatmap(self, function: Callable[[T], Tree[T2]]) -> Tree[T2]:
        return self.from_trees(*[child.flatmap(function) for child in self.children])

    def filter(self, function: Callable[[T], bool]) -> Tree[T]:
        return self.from_trees(*[child.filter(function) for child in self.children])

    def zip(self, *others: Node) -> Node[tuple]:
        others = list(others)

        if not all([isinstance(other, Node) for other in others]):
            raise TypeError("Expected only `Node`s")

        if not all([len(other.children) == len(self.children) for other in others]):
            raise ValueError("Expected all `Node`s to have the same number of children")

        children: list[Tree[tuple[T, ...]]] = []
        for i, child in enumerate(self.children):
            others_child = [other.children[i] for other in others]
            children.append(child.zip(*others_child))
        return Node(*children)

    def __len__(self) -> int:
        return sum([len(child) for child in self.children])

    def __iter__(self) -> Iterator[T]:
        for child in self.children:
            yield from child

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Node) and all(
            a == b for a, b in zip(self.children, other.children)
        )

    def __str__(self) -> str:
        return "<" + ", ".join([str(child) for child in self.children]) + ">"


class Leaf(NonEmptyTree[T]):
    """
    Leaf of a :class:`~torchjd.tree.tree.Tree` that stores a single value.
    """

    def __init__(self, value: T):
        self.value = value

    def depth(self) -> int:
        return 0

    def map(self, function: Callable[[T], T2]) -> Leaf[T2]:
        return Leaf(function(self.value))

    def flatmap(self, function: Callable[[T], Tree[T2]]) -> Tree[T2]:
        return function(self.value)

    def filter(self, function: Callable[[T], bool]) -> Tree[T]:
        if function(self.value):
            return self
        else:
            return EmptyTree()

    def zip(self, *others: Leaf) -> Leaf[tuple]:
        others = list(others)

        if not all([isinstance(other, Leaf) for other in others]):
            raise TypeError("Expected only `Leaf`s")

        return Leaf((self.value,) + tuple(other.value for other in others))

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[T]:
        yield self.value

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Leaf) and other.value == self.value

    def __str__(self) -> str:
        return str(self.value)
