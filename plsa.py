import numpy as np
import math
import re

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        #file = open(self.documents_path)
        #read = file.readlines() 
        #myList = []
        #for line in read:
        #    myList = line.split()
        #    self.documents.append(myList)
        #print(self.documents)
        #self.number_of_documents = len(self.documents) ##MIGHT NEED LEN - 1
        docs = []
        with open('data/test.txt') as f:
            for line in f:
                docs = line.split()
                self.documents.append(docs)

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        #discrete_set = set()
        #for document in self.documents:
        #    for word in document:
        #        discrete_set.add(word)
        #self.vocabulary = list(discrete_set)
        #self.vocabulary_size = len(self.vocabulary)
        # #############################
        # your code here
        # #############################
        data = ""
        x = ""
        fin = []
        with open ('data/test.txt', 'r') as file:
            data = file.read().replace('\n', '')
            x = re.sub('\d', '', data)
        
        uniquewords = set(x.split())
        fin = list(uniquewords)
        self.vocabulary  = fin



    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.
        The term document matrix keeps track of the term frequency of each unique term 
        appeared in each document. So each row corresponds to a document, and each column 
        corresponds to a unique term. You can see in the starter code that, build_term_doc_matrix() 
        is called after build_corpus() and build_vocabulary(), at which stage you already know the 
        number of documents and number of unique terms.
        
        In terms of filling the entries of the matrix, you can either loop over each document and 
        each term in that document, or loop over each document and each unique term in the corpus. 
        Building the matrix for a large corpus can take a while, so the efficiency of your 
        implementation becomes quite important for the second data set (DBLP.txt). In short, 
        there are multiple ways to do it, and you'll need to experiment and find an efficient 
        solution.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        state = {}
        for x in range(0, len(self.documents)):
            state[x] = {}
            for y in range(0, len(self.vocabulary)):
                state[x][y] = self.documents[x].count(self.vocabulary[y])
        self.term_doc_matrix = state
        #print(self.term_doc_matrix)
        
        #pass    # REMOVE THIS


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################

        self.document_topic_prob = np.random.random(size = (self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.random.random(size = (number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)
        #pass    # REMOVE THIS
        
    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        # ############################
        # your code here
        # ############################

        for doc in range(self.number_of_documents):
            for word in range(self.vocabulary_size):
                prob = self.topic_word_prob[:,word] * self.document_topic_prob[doc,:]
                prob = normalize(prob[np.newaxis,:])
                self.topic_prob[d,:, w] = prob
            self.topic_prob = normalize(self.topic_prob[doc,:,:])

        #pass    # REMOVE THIS
            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        # update P(w | z)
        for z in range(number_of_topics):
            for w in range(self.vocabulary_size):
                totals = 0
                for d in range(self.number_of_documents):
                    totals = totals + self.term_doc_matrix[d][w] * self.topic_prob[d, z, w]
                self.topic_word_prob[z][w] = totals
        self.topic_word_prob = normalize(self.topic_word_prob)
        # update P(z | d)
        for d in range(self.number_of_documents):
            for z in range(number_of_topics):
                totals = 0
                for w in range(self.vocabulary_size):
                    totals = totals + self.term_doc_matrix[d][w] * self.topic_prob[d, z, w]
                    print(self.topic_prob[d,z,w])
                self.document_topic_prob[d][z] = totals
        self.document_topic_prob = normalize(self.document_topic_prob)


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        for d_index in range(len(self.documents)):
            for z in range(number_of_topics):
                total = 0
                for w_index in range(self.vocabulary_size):
                    total += self.topic_prob[d_index, z, w_index] * self.topic_word_prob[z,w_index] * self.term_doc_matrix[d_index][w_index]
                total += np.log(total)
        return total

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob =  np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)
            
            current_likelihood = self.likelihoods[len(self.likelihoods) - 1]
            print(current_likelihood)
            if (abs(current_likelihood - self.likelihoods) <= epsilon):
                break



def main():
    documents_path = '/data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
