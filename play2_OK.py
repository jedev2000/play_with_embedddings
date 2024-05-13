# Play with Embeddings
# Jerome C 2024 - jedev2000
# A simple PYTHON program to play with embeddings and words, using command lines, english only
# Totally free usage - no LLM pay plan or API with key required
# Based on Word2vec model, program start with a call to load "word2vec-google-news-300" model
# An external list of words can be used (for some commands), it must be entered in column "A" of file "./samples.xlsx"
# The following libraries must be installed using PIP : gensim, numpy, matplotlib, sklearn, openpyxl, os, sklearn


import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import openpyxl
import os
from sklearn.metrics.pairwise import cosine_similarity


# Load pre-trained Word2Vec model
print("\nLoading word2vec model... ", end="")
word2vec_model = api.load("word2vec-google-news-300")
print ("loaded")


def calculate_embeddings(word):
# function to calculate and return 1 embedding    
    try:
        # Calculate embeddings for the word
        embeddings = word2vec_model[word]
        # print("Embedding : ",embeddings[:5])
        return embeddings
    except KeyError:
        print("Word not found in vocabulary.")
        return None
    

def calculate_word_from_embeddings(word3, embeddings):
# Calculate cosine similarity between the embeddings of the word and all words in the vocabulary
    similarity = np.dot(word2vec_model.vectors, embeddings) / (np.linalg.norm(word2vec_model.vectors, axis=1) * np.linalg.norm(embeddings))
    # Get the index of the most similar word

    # To find closest word
    # Retrieve the word corresponding to the most similar index
    # most_similar_idx = np.argmax(similarity)
    # most_similar_word = word2vec_model.index_to_key[most_similar_idx]

    # To find 1st and 2nd closest words
    indices_tries = np.argsort(similarity)[::-1]
    first_max_index = indices_tries[0]
    second_max_index = indices_tries[1]
    most_similar_word = word2vec_model.index_to_key[first_max_index]
    second_most_similar_word = word2vec_model.index_to_key[second_max_index]
    print("1-- ",most_similar_word)
    print("2-- ",second_most_similar_word)

    # Test to avoid giving answer = input
    if(most_similar_word == word3):
        most_similar_word = second_most_similar_word

    return most_similar_word


def calc_embedding():
# function to calculate and to return 1 embedding without printing it
    word = input("Enter the word to convert : ")
    try:
        # Calculate embeddings for the word
        embeddings = word2vec_model[word]
        # print("Embedding : ",embeddings)
        return embeddings
    except KeyError:
        print("Word not found in vocabulary.")
        return None


def calc_embedding_print():
# function to calculate and to return and to print 1 embedding 
    word = input("Enter the word to convert : ")
    try:
        # Calculate embeddings for the word
        embeddings = word2vec_model[word]
        print("Embedding : ",embeddings)
        return embeddings
    except KeyError:
        print("Word not found in vocabulary.")
        return None
    
def calc_cosine():
# Calculate cosine similarity of 2 embeddings from 2 input words
    word1 = input("Enter word 1 : ")
    try:
        # Calculate embeddings for the word
        embeddings1 = word2vec_model[word1]
    except KeyError:
        print("Word 1 not found in vocabulary.")
        return None

    word2 = input("Enter word 2 : ")
    try:
        # Calculate embeddings for the word
        embeddings2 = word2vec_model[word2]
    except KeyError:
        print("Word 2 not found in vocabulary.")
        return None

    norm_1 = np.linalg.norm(embeddings1)
    norm_2 = np.linalg.norm(embeddings2)

    similarity = np.dot(embeddings1,embeddings2) / (norm_1 * norm_2)

    print("Cosine similarity of ",word1," - ",word2," = ",similarity)
    return


def calc_woperation():
# input 3 words a,b,c and calculate embeddings to perform a-b+c, then show corresponding word to show links between words
    word1 = input("Enter word 1 : ")
    try:
        # Calculate embeddings for the word
        embeddings1 = word2vec_model[word1]
    except KeyError:
        print("Word 1 not found in vocabulary.")
        return None

    word2 = input("Enter word 2 : ")
    try:
        # Calculate embeddings for the word
        embeddings2 = word2vec_model[word2]
    except KeyError:
        print("Word 2 not found in vocabulary.")
        return None

    word3 = input("Enter word 3 : ")
    try:
        # Calculate embeddings for the word
        embeddings1 = word2vec_model[word3]
    except KeyError:
        print("Word 1 not found in vocabulary.")
        return None

    # Calculate embeddings for the word
    embeddings1 = calculate_embeddings(word1)
    embeddings2 = calculate_embeddings(word2)
    embeddings3 = calculate_embeddings(word3)

    embeddings4 = embeddings1 - embeddings2 + embeddings3

    print("Calculating...")
    calculated_word = calculate_word_from_embeddings(word3, embeddings4)
    print(word1," - ",word2," + ", word3, " = ", calculated_word)
    return


def calc_pca():
    # Input N words, calculate embeddings and display them on 2D plan using PCA algorithm
    try:
        size = int(input("Number of words ? "))
    except ValueError:
        print("Incorrect value")
        return None

    embeddings = np.zeros((size, 300))
    names = [''] * size

    for i in range(size):
         print(i+1,"- ",end="")
         word_in = input("enter word : ")
         #print("i=",i)
         try:
             embeddings[i,:]  = word2vec_model[word_in]
             names[i]=word_in
         except KeyError:
            print("Word not found in vocabulary.")
            return None

    # Initialize PCA with 2 components for 2D visualization
    pca = PCA(n_components=2)

    # Fit PCA on the data and transform the embeddings
    embeddings_pca = pca.fit_transform(embeddings)

    # Plot the PCA-transformed embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.5)
    
    # Add name next to each point
    for i, name in enumerate(names):
        plt.text(embeddings_pca[i, 0], embeddings_pca[i, 1], name, fontsize=12, ha='left', va='bottom')
    plt.title('PCA Embeddings Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()
    return


def calc_pca_file():
    # Input N words, calculate embeddings and display on 2D plan using PCA algorithm
    filename = "samples.xlsx"
    filefullname = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

    # Open the Excel file
    wb = openpyxl.load_workbook(filefullname)
    # Select the active sheet (assuming you want to read from the active sheet)
    sheet = wb.active

    # Initialize an empty list to store the values
    values = []

    # Iterate over the rows in the first column (column 'A')
    for cell in sheet['A']:
    # Append the value of each cell to the list
        values.append(cell.value)
    # Close the Excel file
    wb.close()
    # Print the values
    print(values)
    # print(len(values))

    embeddings = np.zeros((len(values),300))
    names = [''] * len(values)

    for i in range(len(values)):
         try:
             # Calculate embeddings for the word
             embeddings[i,:]  = word2vec_model[values[i]]
             names[i]=values[i]
         except KeyError:
            print("Word not found in vocabulary.")
            return None

    # Initialize PCA with 2 components for 2D visualization
    pca = PCA(n_components=2)

    # Fit PCA on the data and transform the embeddings
    embeddings_pca = pca.fit_transform(embeddings)

    # Plot the PCA-transformed embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.5)
    
    # Add name next to each point
    for i, name in enumerate(names):
        plt.text(embeddings_pca[i, 0], embeddings_pca[i, 1], name, fontsize=12, ha='left', va='bottom')
    plt.title('PCA Embeddings Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()
    return


def similarity_search(embeddings_matrix, top_n=5):
    # Perform similarity search to find the top 5 most similar words to the query word.
    # Calculate cosine similarity between the query word embedding and all other word embeddings
    similarities = cosine_similarity(embeddings_matrix[-1, 1:].reshape(1, -1), embeddings_matrix[:, 1:])

    # Get indices of top N most similar words (excluding the query word itself)
    similar_word_indices = np.argsort(-similarities[0])[1:top_n+1]

    # Get the words corresponding to the indices
    similar_words = [embeddings_matrix[idx, 0] for idx in similar_word_indices]

    return similar_words


def calc_closer():
    # Input 1 word, load words from samples.xls, calculate the 5 xlsx words closest to the imput word

    filename = "samples.xlsx"  # Nom du fichier que vous voulez appeler
    filefullname = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

    # Open the Excel file
    wb = openpyxl.load_workbook(filefullname)

    # Select the active sheet (assuming you want to read from the active sheet)
    sheet = wb.active

    # Initialize an empty list to store the values
    values = []

    # Iterate over the rows in the first column (column 'A')
    for cell in sheet['A']:
    # Append the value of each cell to the list
        values.append(cell.value)
    # Close the Excel file
    wb.close()
    print(values)
    #print(len(values))

    embeddings_m = np.zeros((len(values),300))
    embeddings_m2 = np.zeros((len(values),301), dtype=object)

    word = input("Enter word to compare : ")
    try:
        # Calculate embeddings for the word
        embeddings1 = np.zeros((1,300))
        embeddings1 = word2vec_model[word]
        embeddings2 = np.zeros((1,301), dtype=object)
        embeddings2 = np.append(word, embeddings1)
    except KeyError:
        print("Word not found in vocabulary.")
        return None

    for i in range(len(values)):
         try:
             # Calculate embeddings for the word
             embeddings_m[i,:]  = word2vec_model[values[i]]
             embeddings_m2[i,:] = np.append(values[i], embeddings_m[i,:])
         except KeyError:
            print("Word not found in vocabulary.")
            return None

    embeddings_m3 = np.vstack([embeddings_m2, embeddings2])
    top_similar_words = similarity_search(embeddings_m3)
    print(f"Top similar words to '{word}':", top_similar_words)


# Main loop
reponse = "1"
while reponse != "0":
    print ("\n1- Convert a word into an embedding")
    print ("2- Convert 2 words into 2 embeddings and calculate their cosine value")
    print("3- Enter 3 words a,b,c and calculate word value : a - b + c")
    print("4- Display 2D PCA of multiple input words")
    print("5- Display 2D PCA of words in file samples.xlsx")
    print("6- Find closest words in samples.xlsx")
    print("0- Quit\n")

    reponse = int(input("What do you want to do ? "))
    if reponse == 1:
        calc_embedding_print()
    elif reponse == 2:
        calc_cosine()
    elif reponse == 3:
        calc_woperation()
    elif reponse == 4:
        calc_pca()
    elif reponse == 5:
        calc_pca_file()
    elif reponse == 6:
        calc_closer()      
    else:
        break
