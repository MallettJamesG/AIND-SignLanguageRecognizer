import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # BIC={}
        best_model = None
        best_score = float("inf")
        for num_comps in list(range(self.min_n_components, self.max_n_components + 1)):

            logL=0
            try:
                # model = GaussianHMM(n_components=num_comps, n_iter=1000, verbose=False,
                                    # random_state=self.random_state).fit(self.X, self.lengths)
                model = self.base_model(num_comps)
                logL = model.score(self.X, self.lengths)

                # print("Reached after for {} the logL is {}".format(num_comps,logL))

                # Number of parameters is the sum of Transition probs, starting probs,
                # Number of means and variances
                p = num_comps**2 + 2*num_comps*model.n_features - 1
                this_BIC = -2*logL + p*math.log(len(self.X))
                if this_BIC < best_score:
                    best_model = model
                    best_score = this_BIC

            except:
                # print("failure on {} with {} states".format(self.this_word, num_comps))
                # BIC[num_comps] = float('inf')
                pass

        # key_min = min(BIC.keys(), key=(lambda k: BIC[k]))
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        DIC = {}
        Best_DIC = float("-inf")
        DIC_model = None

        for num_comps in list(range(self.min_n_components, self.max_n_components + 1)):

            logL=0
            otherscores = 0
            try:
                model = self.base_model(num_comps)
                this_logL = model.score(self.X, self.lengths)

                for otherword in self.words:
                    if otherword == self.this_word:
                        continue
                    otherX,otherLen = self.hwords[otherword]
                    otherLogL = model.score(otherX,otherLen)
                    otherscores += otherLogL

                thisDIC = this_logL - otherscores/(len(self.words)-1)
                if thisDIC > Best_DIC:
                    DIC_model = model
                    Best_DIC = thisDIC

            except:
                pass

        return DIC_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        raise NotImplementedError
