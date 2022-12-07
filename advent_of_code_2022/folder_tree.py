from __future__ import annotations


class Item:

    def __init__(self, name, parent, *args, **kwargs):
        self.name = name
        self.parent = parent

        assert name == '/' or parent is not None

    @property
    def size(self):
        raise NotImplementedError


class Folder(Item):

    def __init__(self, name, parent, subs=None, *args, **kwargs):
        super().__init__(name, parent)
        assert isinstance(parent, Folder) or parent is None
        if subs is None:
            self.subs = []
        else:
            self.subs = subs

    @property
    def size(self) -> int:
        return sum([s.size for s in self.subs])

    @property
    def subs_dict(self) -> dict:
        return {e.name: e for e in self.subs}

    def add_sub(self, item):
        self.subs.append(item)

    @property
    def parents(self):
        res = []
        p = self.parent
        while p is not None:
            res.append(p)
            p = p.parent
        return res

    def to_str(self, indent=''):
        res = indent + f'- {self.name} (dir)'

        for sub in self.subs:
            res += '\n' + sub.to_str(indent=indent+'| ')
        return res

    def __repr__(self):
        return f'Folder(name={self.name})'

    def __str__(self):
        return self.to_str()


class File(Item):

    def __init__(self, name, parent, size, *args, **kwargs):
        super().__init__(name, parent)
        self._size = size

    @property
    def size(self) -> int:
        return self._size

    def to_str(self, indent=''):
        return indent + f'- {self.name} (file, size={self.size})'

    def __repr__(self):
        return f'File(name={self.name}, size={self.size})'

    def __str__(self):
        return self.to_str()