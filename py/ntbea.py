import numpy as np
import logging
from collections import defaultdict

"""
Author: Chris Bamford
Full code: https://github.com/Bam4d/NTBEA
"""


class SearchSpace(object):
    """
    Inherited by other search space objects
    """

    def __init__(self, name, ndims):
        self._ndims = ndims
        self._name = name

    def get_name(self):
        return self._name

    def get_num_dims(self):
        return self._ndims

    def get_random_point(self):
        raise NotImplementedError()

    def get_size(self):
        raise NotImplementedError()

    def get_dim_size(self, j):
        raise NotImplementedError()

    def get_valid_values_in_dim(self, dim):
        raise NotImplementedError()


class BanditLandscapeModel(object):

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def reset(self):
        raise NotImplementedError()

    def init(self):
        raise NotImplementedError()

    def add_evaluated_point(self, point, fitness):
        raise NotImplementedError()

    def get_mean_estimtate(self, point):
        raise NotImplementedError()

    def get_exploration_estimate(self, point):
        raise NotImplementedError()


class NTupleLandscape(BanditLandscapeModel):
    """
    The N-tuple landscape implementation
    """

    def __init__(self, search_space, tuple_config=None, ucb_epsilon=0.5):
        super(NTupleLandscape, self).__init__('N-Tuple Bandit Landscape')
        self._logger = logging.getLogger('NTupleLandscape')

        # If we dont have a tuple config, we just create a default tuple config, the 1-tuples and N-tuples
        if tuple_config is None:
            tuple_config = [1, search_space.get_num_dims()]

        self._tuple_config = set(tuple_config)
        self._tuples = list()
        self._ndims = search_space.get_num_dims()

        self._sampled_points = set()

        self._ucb_epsilon = ucb_epsilon

        self.reset()

    def reset(self):
        self._tuple_stats = defaultdict(
            lambda: defaultdict(
                lambda: {
                    'n': 0,
                    'min': 0.0,
                    'max': 0.0,
                    'sum': 0.0,
                    'sum_squared': 0.0
                }
            )
        )

    def get_tuple_combinations(self, r, ndims):
        """
        Get the unique combinations of tuples for the n-tuple landscape
        :param r: the 'n' value of this tuple
        :param ndims: the number of dimensions in the search space
        :return:
        """
        return self._get_unique_combinations(0, r, range(0, ndims))

    def _get_unique_combinations(self, idx, r, source_array):

        result = []
        for i in range(idx, len(source_array)):
            if r - 1 > 0:
                next_level = self._get_unique_combinations(i + 1, r - 1, source_array)
                for x in next_level:
                    value = [source_array[i]]
                    value.extend(x)
                    result.append(value)

            else:
                result.append([source_array[i]])

        return result

    def init(self):
        """
        Create the index combinations for each of the n-tuples
        """
        # Create all possible tuples for each
        for n in self._tuple_config:
            n_tuples = [tup for tup in self.get_tuple_combinations(n, self._ndims)]
            self._tuples.extend(n_tuples)
            self._logger.debug('Added %d-tuples: %s' % (n, n_tuples))

        self._logger.info('Tuple Landscape Size: %d' % len(self._tuples))

    def add_evaluated_point(self, point, fitness):
        """
        Add a point and it's fitness to the tuple landscape
        """

        self._sampled_points.add(tuple(point))

        for tup in self._tuples:
            # The search space value is the values given by applying the tuple to the search space.
            # This is used to index the stats at that point for the particular tuple in question
            search_space_value = point[tup]

            # Use 'totals' as a key to store summary data of the tuple
            self._tuple_stats[tuple(tup)]['totals']['n'] += 1

            search_space_tuple_stats = self._tuple_stats[tuple(tup)][tuple(search_space_value)]
            search_space_tuple_stats['n'] += 1
            search_space_tuple_stats['max'] = np.maximum(search_space_tuple_stats['max'], fitness)
            search_space_tuple_stats['min'] = np.minimum(search_space_tuple_stats['max'], fitness)
            search_space_tuple_stats['sum'] += fitness
            search_space_tuple_stats['sum_squared'] += fitness ** 2

    def get_mean_estimtate(self, point):
        """
        Iterate over all the tuple stats we have stored for this point and sum the means and the number
        of stats we have found.
        Finally the sum of the means divided by the total number of stats found is returned
        """
        summ = 0
        tuple_count = 0
        vals = []
        for tup in self._tuples:
            search_space_value = point[tup]
            tuple_stats = self._tuple_stats[tuple(tup)][tuple(search_space_value)]
            if tuple_stats['n'] > 0:
                summ += tuple_stats['sum'] / tuple_stats['n']
                tuple_count += 1
                vals.append(tuple_stats['sum'] / tuple_stats['n'])

        if tuple_count == 0:
            return 0

        vals = np.array(vals)
        return np.mean(vals), np.std(vals)  # summ / tuple_count

    def get_exploration_estimate(self, point):
        """
        Calculate the average of the exploration across all tuples of the exploration
        """

        summ = 0
        tuple_count = len(self._tuples)

        for tup in self._tuples:
            search_space_value = point[tup]
            tuple_stats = self._tuple_stats[tuple(tup)]
            search_space_tuple_stats = tuple_stats[tuple(search_space_value)]
            if search_space_tuple_stats['n'] == 0:
                n = tuple_stats['totals']['n']
                summ += np.sqrt(np.log(1 + n) / self._ucb_epsilon)
            else:
                n = search_space_tuple_stats['n']
                summ += np.sqrt(np.log(1 + n) / (n + self._ucb_epsilon))

        return summ / tuple_count

    def get_best_sampled(self):

        current_best_mean = 0
        current_best_point = None
        for point in self._sampled_points:
            mean, _ = self.get_mean_estimtate(np.array(point))

            if mean > current_best_mean:
                current_best_mean = mean
                current_best_point = point

        return current_best_point

    def get_average_value(self):
        vals = []
        for point in self._sampled_points:
            mean, _ = self.get_mean_estimtate(np.array(point))
            vals.append(mean)
        return np.mean(vals)
