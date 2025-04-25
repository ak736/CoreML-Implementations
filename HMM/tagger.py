import numpy as np
from hmm import HMM


def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)

    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index
    #   - from a tag to its index
    # The order you index the word/tag does not matter,
    # as long as the indices are 0, 1, 2, ...
    ###################################################

    # Build word-to-index dictionary
    for idx, word in enumerate(unique_words.keys()):
        word2idx[word] = idx

    # Build tag-to-index dictionary
    for idx, tag in enumerate(tags):
        tag2idx[tag] = idx

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if
    #   "divided by zero" is encountered, set the entry
    #   to be zero.
    ###################################################

    # Count the initial state occurrences and transitions
    initial_state_counts = np.zeros(S)
    transition_counts = np.zeros((S, S))
    emission_counts = np.zeros((S, len(unique_words)))
    state_counts = np.zeros(S)

    # Process each sentence in the training data
    for line in train_data:
        # Count initial state (first tag in the sentence)
        if line.length > 0:
            initial_tag = line.tags[0]
            initial_state_counts[tag2idx[initial_tag]] += 1

            # Count transitions and emissions
            for t in range(line.length):
                curr_tag = line.tags[t]
                curr_word = line.words[t]
                curr_tag_idx = tag2idx[curr_tag]

                # Count state occurrence
                state_counts[curr_tag_idx] += 1

                # Count emissions
                if curr_word in word2idx:
                    emission_counts[curr_tag_idx, word2idx[curr_word]] += 1

                # Count transitions (not for the last tag)
                if t < line.length - 1:
                    next_tag = line.tags[t+1]
                    next_tag_idx = tag2idx[next_tag]
                    transition_counts[curr_tag_idx, next_tag_idx] += 1

    # Compute pi (initial state probabilities)
    total_sentences = sum(initial_state_counts)
    if total_sentences > 0:
        pi = initial_state_counts / total_sentences

    # Compute A (transition probabilities)
    for i in range(S):
        if state_counts[i] > 0:
            A[i, :] = transition_counts[i, :] / state_counts[i]

    # Compute B (emission probabilities)
    for i in range(S):
        if state_counts[i] > 0:
            B[i, :] = emission_counts[i, :] / state_counts[i]

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################

    for sentence in test_data:
        # Check for unseen words and update model.obs_dict and model.B if necessary
        for word in sentence.words:
            if word not in model.obs_dict:
                # Add the new word to the observation dictionary
                new_word_idx = len(model.obs_dict)
                model.obs_dict[word] = new_word_idx

                # Expand the emission matrix B
                old_B = model.B
                model.B = np.zeros((old_B.shape[0], old_B.shape[1] + 1))
                model.B[:, :old_B.shape[1]] = old_B

                # Set probability of the new word to 1e-6 for all states
                model.B[:, new_word_idx] = 1e-6

        # Use Viterbi algorithm to find the most likely tag sequence
        viterbi_path = model.viterbi(sentence.words)
        tagging.append(viterbi_path)

    return tagging

# DO NOT MODIFY BELOW


def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
