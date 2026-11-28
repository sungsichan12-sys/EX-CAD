# SPDX-License-Identifier: GPL-2.0-or-later
"""Match islands and cavities between successive planar sections of a loft."""

from collections import defaultdict

from mathutils import Vector

from .sketch_profiles import contains, cross, signed_area


def _assignment(cost):
    """Minimum-sum one-to-one assignment of equally many section boundaries."""
    n = len(cost)
    if n == 1:
        return [0]
    # Augment one row at a time using shortest alternating paths. Prices keep
    # every unused edge's reduced cost nonnegative after each augmentation.
    row_price, col_price = [0.0] * (n + 1), [0.0] * (n + 1)
    match = [0] * (n + 1)
    for row in range(1, n + 1):
        match[0] = row
        slack, previous = [float("inf")] * (n + 1), [0] * (n + 1)
        used = [False] * (n + 1)
        column = 0
        while match[column]:
            used[column] = True
            current = match[column]
            next_column = 0
            for other in range(1, n + 1):
                if used[other]:
                    continue
                price = cost[current - 1][other - 1] - row_price[current] - col_price[other]
                if price < slack[other]:
                    slack[other], previous[other] = price, column
                if not next_column or slack[other] < slack[next_column]:
                    next_column = other
            step = slack[next_column]
            for other in range(n + 1):
                if used[other]:
                    row_price[match[other]] += step
                    col_price[other] -= step
                else:
                    slack[other] -= step
            column = next_column
        while column:
            parent = previous[column]
            match[column] = match[parent]
            column = parent
    result = [0] * n
    for column in range(1, n + 1):
        result[match[column] - 1] = column - 1
    return result


class _Section:
    def __init__(self, doc, loops):
        areas = [abs(signed_area(loop)) for loop in loops]
        self.centers, children = [], defaultdict(list)
        for i, loop in enumerate(loops):
            # Area centroid is independent of how finely each edge was sampled.
            center = sum(
                ((a + b) * cross(a, b) for a, b in zip(loop, loop[1:] + loop[:1])), Vector((0, 0))
            ) / (6 * signed_area(loop))
            self.centers.append(Vector(doc.plane_to_world(*center, 0)))
            enclosing = [j for j, other in enumerate(loops) if areas[j] > areas[i] and contains(loop[0], other)]
            parent = min(enclosing, key=areas.__getitem__) if enclosing else None
            children[parent].append(i)

        def shape(index):
            return tuple(sorted(shape(child) for child in children[index]))

        self.families = defaultdict(lambda: defaultdict(list))
        for parent, indices in list(children.items()):
            for index in indices:
                self.families[parent][shape(index)].append(index)


def ordered_sections(docs, sections):
    """Preserve containment and minimize travel, even when hole diameters exchange order."""
    if all(len(loops) == 1 for loops in sections):
        return sections
    ordered = []
    previous = None
    for doc, loops in zip(docs, sections):
        section = _Section(doc, loops)
        if previous is None:
            order = list(range(len(loops)))
        else:
            matches = {}
            pending = [(None, None)]
            while pending:
                old_parent, new_parent = pending.pop()
                old_family, new_family = previous.families[old_parent], section.families[new_parent]
                if old_family.keys() != new_family.keys():
                    raise ValueError("Loft sections have different island or hole nesting")
                for shape, old in old_family.items():
                    new = new_family[shape]
                    if len(old) != len(new):
                        raise ValueError("Loft sections have different island or hole counts")
                    costs = [[(previous.centers[i] - section.centers[j]).length_squared for j in new] for i in old]
                    for i, choice in zip(old, _assignment(costs)):
                        matches[i] = new[choice]
                        pending.append((i, new[choice]))
            order = [matches[i] for i in order]
        ordered.append([loops[i] for i in order])
        previous = section
    return ordered
