# -*- coding: utf-8 -*-

"""
    File name: gibbs_sampler.py
    Description: A re-implementation of the Gibbs sampler for LDA
    Author: Python: Roman Pogodin, MATLAB (original): Yee Whye Teh and Maneesh Sahani
    Date created: October 2018
    Python version: 3.6 and above
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln
import matplotlib.pyplot as plt

# Configure matplotlib for better aesthetics
plt.rcParams.update({'font.size': 12})
plt.style.use('seaborn-darkgrid')  # Optional: Use a seaborn-like style


class GibbsSampler:
    def __init__(self, n_docs, n_topics, n_words, alpha, beta, random_seed=None):
        """
        :param n_docs:          Number of documents
        :param n_topics:        Number of topics
        :param n_words:         Number of words in vocabulary
        :param alpha:           Dirichlet parameter on topic mixing proportions
        :param beta:            Dirichlet parameter on topic word distributions
        :param random_seed:     Random seed for reproducibility
        """
        self.n_docs = n_docs
        self.n_topics = n_topics
        self.n_words = n_words
        self.alpha = alpha
        self.beta = beta
        self.rand_gen = np.random.RandomState(random_seed)

        self.docs_words = np.zeros((self.n_docs, self.n_words), dtype=int)
        self.docs_words_test = None
        self.loglike = None
        self.loglike_test = None
        self.do_test = False

        self.A_dk = np.zeros((self.n_docs, self.n_topics), dtype=int)  # Words per document per topic
        self.B_kw = np.zeros((self.n_topics, self.n_words), dtype=int)  # Words per topic

        self.theta = np.ones((self.n_docs, self.n_topics)) / self.n_topics  # Topic distributions per document
        self.phi = np.ones((self.n_topics, self.n_words)) / self.n_words  # Word distributions per topic

        self.topics_space = np.arange(self.n_topics)
        self.topic_doc_words_distr = np.zeros((self.n_docs, self.n_topics, self.n_words))

    def init_sampling(self, docs_words, docs_words_test=None,
                      theta=None, phi=None, n_iter=0, save_loglike=False):
        assert docs_words.shape == (self.n_docs, self.n_words), \
            f"docs_words shape={docs_words.shape} must be ({self.n_docs}, {self.n_words})"

        self.docs_words = docs_words
        self.docs_words_test = docs_words_test

        self.do_test = docs_words_test is not None

        if save_loglike:
            self.loglike = np.zeros(n_iter)
            if self.do_test:
                self.loglike_test = np.zeros(n_iter)

        # Reset counts
        self.A_dk.fill(0)
        self.B_kw.fill(0)

        if self.do_test:
            self.A_dk_test = np.zeros((self.n_docs, self.n_topics), dtype=int)
            self.B_kw_test = np.zeros((self.n_topics, self.n_words), dtype=int)
            self.A_dk_test.fill(0)
            self.B_kw_test.fill(0)

        # Initialize parameters
        self.init_params(theta, phi)

    def init_params(self, theta=None, phi=None):
        if theta is None:
            self.theta = np.ones((self.n_docs, self.n_topics)) / self.n_topics
        else:
            self.theta = theta.copy()

        if phi is None:
            self.phi = np.ones((self.n_topics, self.n_words)) / self.n_words
        else:
            self.phi = phi.copy()

        self.update_topic_doc_words()
        self.sample_counts()

    def run(self, docs_words, docs_words_test=None,
            n_iter=100, theta=None, phi=None, save_loglike=False):
        """
        Runs the Gibbs sampler.

        :param docs_words:        Training data matrix [n_docs, n_words]
        :param docs_words_test:   Test data matrix [n_docs, n_words]
        :param n_iter:            Number of iterations
        :param theta:             Initial theta (optional)
        :param phi:               Initial phi (optional)
        :param save_loglike:      Whether to save log-likelihoods
        :return:                  Tuple containing topic distributions and word distributions
        """
        self.init_sampling(docs_words, docs_words_test,
                           theta, phi, n_iter, save_loglike)

        for iteration in range(n_iter):
            self.update_params()

            if save_loglike:
                self.update_loglike(iteration)

            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iteration {iteration + 1}/{n_iter} completed.")

        return self.to_return_from_run()

    def to_return_from_run(self):
        return self.topic_doc_words_distr, self.theta, self.phi

    def update_params(self):
        """
        Samples theta and phi from their posterior distributions and updates counts.
        """
        # Sample theta for each document from Dirichlet(alpha + A_dk)
        self.theta = self.rand_gen.dirichlet(self.alpha + self.A_dk, size=self.n_docs)

        # Sample phi for each topic from Dirichlet(beta + B_kw)
        self.phi = self.rand_gen.dirichlet(self.beta + self.B_kw, size=self.n_topics)

        # Update the topic-word distribution
        self.update_topic_doc_words()

        # Sample counts based on the updated distributions
        self.sample_counts()

    def update_topic_doc_words(self):
        """
        Computes the distribution of topics for each word in each document.
        """
        # Compute unnormalized probabilities
        topic_probs = self.theta[:, :, np.newaxis] * self.phi[np.newaxis, :, :]
        # Normalize across topics (axis=1)
        topic_probs /= np.sum(topic_probs, axis=1, keepdims=True) + 1e-12
        self.topic_doc_words_distr = topic_probs

    def sample_counts(self):
        """
        Samples topic assignments for each word in each document based on current theta and phi.
        Updates the counts A_dk and B_kw accordingly.
        """
        self.A_dk.fill(0)
        self.B_kw.fill(0)

        if self.do_test:
            self.A_dk_test.fill(0)
            self.B_kw_test.fill(0)

        for d in range(self.n_docs):
            for w in range(self.n_words):
                count = self.docs_words[d, w]
                if count > 0:
                    prob = self.topic_doc_words_distr[d, :, w]
                    prob /= np.sum(prob)  # Normalize to ensure it's a probability distribution

                    # Sample topic assignments using multinomial
                    topics = self.rand_gen.multinomial(count, prob)
                    self.A_dk[d] += topics
                    self.B_kw[:, w] += topics

                if self.do_test:
                    count_test = self.docs_words_test[d, w]
                    if count_test > 0:
                        prob_test = self.topic_doc_words_distr[d, :, w]
                        prob_test /= np.sum(prob_test)

                        topics_test = self.rand_gen.multinomial(count_test, prob_test)
                        self.A_dk_test[d] += topics_test
                        self.B_kw_test[:, w] += topics_test

    def update_loglike(self, iteration):
        """
        Updates the log-likelihood of the data.

        :param iteration: Current iteration number
        """
        # Compute log-likelihood for training data
        ll = 0

        # Document-Topic distributions
        ll += np.sum(gammaln(self.alpha + self.A_dk) - gammaln(self.alpha))
        ll -= np.sum(gammaln(np.sum(self.alpha + self.A_dk, axis=1)) - gammaln(np.sum(self.alpha)))

        # Topic-Word distributions
        ll += np.sum(gammaln(self.beta + self.B_kw) - gammaln(self.beta))
        ll -= np.sum(gammaln(np.sum(self.beta + self.B_kw, axis=1)) - gammaln(np.sum(self.beta)))

        self.loglike[iteration] = ll

        # Compute log-likelihood for test data if necessary
        if self.do_test:
            ll_test = 0

            # Document-Topic distributions
            ll_test += np.sum(gammaln(self.alpha + self.A_dk_test) - gammaln(self.alpha))
            ll_test -= np.sum(gammaln(np.sum(self.alpha + self.A_dk_test, axis=1)) - gammaln(np.sum(self.alpha)))

            # Topic-Word distributions
            ll_test += np.sum(gammaln(self.beta + self.B_kw_test) - gammaln(self.beta))
            ll_test -= np.sum(gammaln(np.sum(self.beta + self.B_kw_test, axis=1)) - gammaln(np.sum(self.beta)))

            self.loglike_test[iteration] = ll_test

    def get_loglike(self):
        """Returns log-likelihood at each iteration."""
        if self.do_test:
            return self.loglike, self.loglike_test
        else:
            return self.loglike


class GibbsSamplerCollapsed(GibbsSampler):
    def __init__(self, n_docs, n_topics, n_words, alpha, beta, random_seed=None):
        """
        :param n_docs:          Number of documents
        :param n_topics:        Number of topics
        :param n_words:         Number of words in vocabulary
        :param alpha:           Dirichlet parameter on topic mixing proportions
        :param beta:            Dirichlet parameter on topic word distributions
        :param random_seed:     Random seed for reproducibility
        """
        super().__init__(n_docs, n_topics, n_words, alpha, beta, random_seed)

        # Topics assigned to each (doc, word)
        self.doc_word_samples = np.empty((self.n_docs, self.n_words), dtype=object)
        self.doc_word_samples_test = np.empty((self.n_docs, self.n_words), dtype=object)

    def init_params(self, theta=None, phi=None):
        # Initialize topic assignments uniformly
        for d in range(self.n_docs):
            for w in range(self.n_words):
                count = self.docs_words[d, w]
                if count > 0:
                    sampled_topics = self.rand_gen.randint(0, self.n_topics, size=count)
                    self.doc_word_samples[d, w] = sampled_topics.copy()
                    for k in sampled_topics:
                        self.A_dk[d, k] += 1
                        self.B_kw[k, w] += 1
                else:
                    self.doc_word_samples[d, w] = np.array([], dtype=int)

                if self.do_test:
                    count_test = self.docs_words_test[d, w]
                    if count_test > 0:
                        sampled_topics_test = self.rand_gen.randint(0, self.n_topics, size=count_test)
                        self.doc_word_samples_test[d, w] = sampled_topics_test.copy()
                        for k in sampled_topics_test:
                            self.A_dk_test[d, k] += 1
                            self.B_kw_test[k, w] += 1
                    else:
                        self.doc_word_samples_test[d, w] = np.array([], dtype=int)

    def update_params(self):
        """
        Performs one iteration of collapsed Gibbs sampling.
        """
        for d in range(self.n_docs):
            for w in range(self.n_words):
                count = self.docs_words[d, w]
                if count > 0:
                    for i in range(count):
                        # Current topic assignment
                        current_topic = self.doc_word_samples[d, w][i]

                        # Decrement counts
                        self.A_dk[d, current_topic] -= 1
                        self.B_kw[current_topic, w] -= 1

                        # Compute conditional distribution
                        topic_probs = (self.A_dk[d] + self.alpha) * (self.B_kw[:, w] + self.beta)
                        topic_probs /= (np.sum(self.B_kw, axis=1) + self.n_words * self.beta)
                        topic_probs /= np.sum(topic_probs)  # Normalize

                        # Sample new topic
                        new_topic = self.rand_gen.choice(self.topics_space, p=topic_probs)
                        self.doc_word_samples[d, w][i] = new_topic

                        # Increment counts
                        self.A_dk[d, new_topic] += 1
                        self.B_kw[new_topic, w] += 1

                if self.do_test:
                    count_test = self.docs_words_test[d, w]
                    if count_test > 0:
                        for i in range(count_test):
                            current_topic = self.doc_word_samples_test[d, w][i]

                            # Decrement counts
                            self.A_dk_test[d, current_topic] -= 1
                            self.B_kw_test[current_topic, w] -= 1

                            # Compute conditional distribution
                            topic_probs_test = (self.A_dk_test[d] + self.alpha) * (self.B_kw_test[:, w] + self.beta)
                            topic_probs_test /= (np.sum(self.B_kw_test, axis=1) + self.n_words * self.beta)
                            topic_probs_test /= np.sum(topic_probs_test)  # Normalize

                            # Sample new topic
                            new_topic_test = self.rand_gen.choice(self.topics_space, p=topic_probs_test)
                            self.doc_word_samples_test[d, w][i] = new_topic_test

                            # Increment counts
                            self.A_dk_test[d, new_topic_test] += 1
                            self.B_kw_test[new_topic_test, w] += 1

    def update_loglike(self, iteration):
        """
        Updates the log-likelihood of the data.

        :param iteration: Current iteration number
        """
        # Compute log-likelihood for training data
        ll = 0

        # Document-Topic distributions
        for d in range(self.n_docs):
            ll += gammaln(self.alpha * self.n_topics) - self.n_topics * gammaln(self.alpha)
            ll += np.sum(gammaln(self.A_dk[d] + self.alpha))
            ll -= gammaln(np.sum(self.A_dk[d] + self.alpha))

        # Topic-Word distributions
        for k in range(self.n_topics):
            ll += gammaln(self.beta * self.n_words) - self.n_words * gammaln(self.beta)
            ll += np.sum(gammaln(self.B_kw[k] + self.beta))
            ll -= gammaln(np.sum(self.B_kw[k] + self.beta))

        self.loglike[iteration] = ll

        # Compute log-likelihood for test data if necessary
        if self.do_test:
            ll_test = 0

            # Document-Topic distributions
            for d in range(self.n_docs):
                ll_test += gammaln(self.alpha + self.A_dk_test[d])
                ll_test -= gammaln(np.sum(self.alpha + self.A_dk_test[d]))

            # Topic-Word distributions
            for k in range(self.n_topics):
                ll_test += gammaln(self.beta + self.B_kw_test[k])
                ll_test -= gammaln(np.sum(self.beta + self.B_kw_test[k]))

            self.loglike_test[iteration] = ll_test

    def to_return_from_run(self):
        return self.doc_word_samples

    def get_theta_phi(self):
        """
        Computes theta and phi based on the current counts.

        :return: Tuple of (theta, phi)
        """
        theta = (self.A_dk + self.alpha)
        theta /= np.sum(theta, axis=1)[:, np.newaxis]

        phi = (self.B_kw + self.beta)
        phi /= np.sum(phi, axis=1)[:, np.newaxis]

        return theta, phi


def read_data(filename):
    """
    Reads the text data and splits into train/test.

    :param filename: Path to the data file
    :return: Tuple of training and test data matrices
    """
    data = pd.read_csv(filename, dtype=int, sep=' ', names=['doc', 'word', 'train', 'test'])

    n_docs = data['doc'].max()
    n_words = data['word'].max()

    docs_words_train = np.zeros((n_docs, n_words), dtype=int)
    docs_words_test = np.zeros((n_docs, n_words), dtype=int)

    for row in data.itertuples(index=False):
        d = row.doc - 1  # Assuming docs are 1-indexed
        w = row.word - 1  # Assuming words are 1-indexed
        docs_words_train[d, w] = row.train
        docs_words_test[d, w] = row.test

    return docs_words_train, docs_words_test


def main():
    print('Running toyexample.data with the standard sampler')

    # Replace './toyexample.data' with the correct path to your data file
    docs_words_train, docs_words_test = read_data('./toyexample.data')
    n_docs, n_words = docs_words_train.shape
    n_topics = 3
    alpha = 1.0
    beta = 1.0
    random_seed = 0

    sampler = GibbsSampler(n_docs=n_docs, n_topics=n_topics, n_words=n_words,
                           alpha=alpha, beta=beta, random_seed=random_seed)

    topic_doc_words_distr, theta, phi = sampler.run(docs_words_train, docs_words_test,
                                                    n_iter=200, save_loglike=True)

    print("Phi (topic-word distributions):")
    print(phi * (phi > 1e-2))  # Display words with significant probabilities

    like_train, like_test = sampler.get_loglike()

    plt.figure(figsize=(15, 6))
    plt.plot(like_train, label='Train')
    plt.plot(like_test, label='Test')
    plt.ylabel('Log Likelihood')
    plt.xlabel('Iteration')
    plt.title('Standard Gibbs Sampler on Toy Example')
    plt.legend()
    plt.show()

    print('Running toyexample.data with the collapsed sampler')

    sampler_collapsed = GibbsSamplerCollapsed(n_docs=n_docs, n_topics=n_topics, n_words=n_words,
                                              alpha=alpha, beta=beta, random_seed=random_seed)

    doc_word_samples = sampler_collapsed.run(docs_words_train, docs_words_test,
                                             n_iter=200, save_loglike=True)

    theta_collapsed, phi_collapsed = sampler_collapsed.get_theta_phi()

    print("Phi (topic-word distributions) from collapsed Gibbs sampler:")
    print(phi_collapsed * (phi_collapsed > 1e-2))  # Display words with significant probabilities

    like_train_collapsed, like_test_collapsed = sampler_collapsed.get_loglike()

    plt.figure(figsize=(15, 6))
    plt.plot(like_train_collapsed, label='Train (Collapsed)')
    plt.plot(like_test_collapsed, label='Test (Collapsed)')
    plt.ylabel('Log Likelihood')
    plt.xlabel('Iteration')
    plt.title('Collapsed Gibbs Sampler on Toy Example')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
