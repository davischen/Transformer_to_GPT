from huggingface_hub import login

from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import zipfile
import requests
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import random


class TextSimilarityModel:
    def __init__(self, corpus_name, rel_name, model_name='all-MiniLM-L6-v2', top_k=10):
        """
        Initialize the model with datasets and pre-trained sentence transformer.
        """
        self.model = SentenceTransformer(model_name)
        self.corpus_name = corpus_name
        self.rel_name = rel_name
        self.top_k = top_k
        self.load_data()

    def load_data(self):
        """
        Load and filter datasets based on test queries and documents.
        """
        # Load query and document datasets
        dataset_queries = load_dataset(self.corpus_name, "queries")
        dataset_docs = load_dataset(self.corpus_name, "corpus")

        # Extract queries and documents
        self.queries = dataset_queries["queries"]["text"]
        self.query_ids = dataset_queries["queries"]["_id"]
        self.documents = dataset_docs["corpus"]["text"]
        self.document_ids = dataset_docs["corpus"]["_id"]

        # Filter queries and documents and build relevant queries and documents mapping based on test set
        test_qrels = load_dataset(self.rel_name)["test"]
        self.filtered_test_query_ids = set(test_qrels["query-id"])
        self.filtered_test_doc_ids = set(test_qrels["corpus-id"])

        self.test_queries = [q for qid, q in zip(self.query_ids, self.queries) if qid in self.filtered_test_query_ids]
        self.test_query_ids = [qid for qid in self.query_ids if qid in self.filtered_test_query_ids]
        self.test_documents = [doc for did, doc in zip(self.document_ids, self.documents) if
                               did in self.filtered_test_doc_ids]
        self.test_document_ids = [did for did in self.document_ids if did in self.filtered_test_doc_ids]

        self.test_query_id_to_relevant_doc_ids = {qid: [] for qid in self.test_query_ids}
        for qid, doc_id in zip(test_qrels["query-id"], test_qrels["corpus-id"]):
            if qid in self.test_query_id_to_relevant_doc_ids:
                self.test_query_id_to_relevant_doc_ids[qid].append(doc_id)

        ## Code Below this is used for creating the training set
        # Build query and document id to text mapping
        self.query_id_to_text = {query_id: query for query_id, query in zip(self.query_ids, self.queries)}
        self.document_id_to_text = {document_id: document for document_id, document in
                                    zip(self.document_ids, self.documents)}

        # Build relevant queries and documents mapping based on train set
        train_qrels = load_dataset(self.rel_name)["train"]
        self.train_query_id_to_relevant_doc_ids = {qid: [] for qid in train_qrels["query-id"]}

        for qid, doc_id in zip(train_qrels["query-id"], train_qrels["corpus-id"]):
            if qid in self.train_query_id_to_relevant_doc_ids:
                # Append the document ID to the relevant doc mapping
                self.train_query_id_to_relevant_doc_ids[qid].append(doc_id)

        # Filter queries and documents and build relevant queries and documents mapping based on validation set
        # TODO Put your code here.

        validation_qrels = load_dataset(self.rel_name)["validation"]

        self.filtered_validation_query_ids = set(validation_qrels["query-id"])
        self.filtered_validation_doc_ids = set(validation_qrels["corpus-id"])

        self.validation_queries = [q for qid, q in zip(self.query_ids, self.queries) if
                                   qid in self.filtered_validation_query_ids]
        self.validation_query_ids = [qid for qid in self.query_ids if qid in self.filtered_validation_query_ids]
        self.validation_documents = [doc for did, doc in zip(self.document_ids, self.documents) if
                                     did in self.filtered_validation_doc_ids]
        self.validation_document_ids = [did for did in self.document_ids if did in self.filtered_validation_doc_ids]

        self.validation_query_id_to_relevant_doc_ids = {qid: [] for qid in self.validation_query_ids}
        for qid, doc_id in zip(validation_qrels["query-id"], validation_qrels["corpus-id"]):
            if qid in self.validation_query_id_to_relevant_doc_ids:
                self.validation_query_id_to_relevant_doc_ids[qid].append(doc_id)

    # Task 1: Encode Queries and Documents (10 Pts)

    def encode_with_glove(self, glove_file_path: str, sentences: list[str]) -> list[np.ndarray]:

        """
        # Inputs:
            - glove_file_path (str): Path to the GloVe embeddings file (e.g., "glove.6B.50d.txt").
            - sentences (list[str]): A list of sentences to encode.

        # Output:
            - list[np.ndarray]: A list of sentence embeddings

        (1) Encodes sentences by averaging GloVe 50d vectors of words in each sentence.
        (2) Return a sequence of embeddings of the sentences.
        Download the glove vectors from here.
        https://nlp.stanford.edu/data/glove.6B.zip
        Handle unknown words by using zero vectors
        """
        # TODO Put your code here.

        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        save_path = "glove.6B.zip"
        extract_path = "glove.6B"

        if not os.path.exists(save_path):
            print("Downloading GloVe embeddings...")
            response = requests.get(url, stream=True)
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print("Download complete!")

        if not os.path.exists(extract_path):
            print("Extracting GloVe embeddings...")
            with zipfile.ZipFile(save_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            print("GloVe embeddings extracted successfully!")

        glove_embeddings = {}
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                glove_embeddings[word] = vector

        embedding_dim = len(next(iter(glove_embeddings.values())))

        sentence_embeddings = []
        for sentence in sentences:
            words = sentence.lower().split()
            word_embeddings = []
            for word in words:
                if word in glove_embeddings:
                    word_embeddings.append(glove_embeddings[word])
                else:
                    word_embeddings.append(np.zeros(embedding_dim))

            if word_embeddings:
                sentence_embedding = np.mean(word_embeddings, axis=0)
            else:
                sentence_embedding = np.zeros(embedding_dim)

            sentence_embeddings.append(sentence_embedding)

        return sentence_embeddings

    # Task 2: Calculate Cosine Similarity and Rank Documents (20 Pts)

    def rank_documents(self, encoding_method: str = 'sentence_transformer') -> None:
        """
         # Inputs:
            - encoding_method (str): The method used for encoding queries/documents.
                             Options: ['glove', 'sentence_transformer'].

        # Output:
            - None (updates self.query_id_to_ranked_doc_ids with ranked document IDs).

        (1) Compute cosine similarity between each document and the query
        (2) Rank documents for each query and save the results in a dictionary "query_id_to_ranked_doc_ids"
            This will be used in "mean_average_precision"
            Example format {2: [125, 673], 35: [900, 822]}
        """
        if encoding_method == 'glove':
            query_embeddings = self.encode_with_glove("glove.6B/glove.6B.50d.txt", self.queries)
            document_embeddings = self.encode_with_glove("glove.6B/glove.6B.50d.txt", self.documents)
        elif encoding_method == 'sentence_transformer':
            query_embeddings = self.model.encode(self.queries)
            document_embeddings = self.model.encode(self.documents)
        else:
            raise ValueError("Invalid encoding method. Choose 'glove' or 'sentence_transformer'.")

        # TODO Put your code here.
        ###########################################################################
        # define a dictionary to store the ranked documents for each query

        self.query_id_to_ranked_doc_ids = {}

        for query_id, query_embedding in zip(self.query_ids, query_embeddings):
            similarities = cosine_similarity([query_embedding], document_embeddings)[0]
            ranked_doc_indices = np.argsort(similarities)[::-1][:self.top_k]
            ranked_doc_ids = [self.document_ids[idx] for idx in ranked_doc_indices]
            self.query_id_to_ranked_doc_ids[query_id] = ranked_doc_ids
        ###########################################################################

    @staticmethod
    def average_precision(relevant_docs: list[str], candidate_docs: list[str]) -> float:
        """
        # Inputs:
            - relevant_docs (list[str]): A list of document IDs that are relevant to the query.
            - candidate_docs (list[str]): A list of document IDs ranked by the model.

        # Output:
            - float: The average precision score

        Compute average precision for a single query.
        """
        y_true = [1 if doc_id in relevant_docs else 0 for doc_id in candidate_docs]
        precisions = [np.mean(y_true[:k + 1]) for k in range(len(y_true)) if y_true[k]]
        return np.mean(precisions) if precisions else 0

    # Task 3: Calculate Evaluate System Performance (10 Pts)

    def mean_average_precision(self) -> float:
        """
        # Inputs:
            - None (uses ranked documents stored in self.query_id_to_ranked_doc_ids).

        # Output:
            - float: The MAP score, computed as the mean of all average precision scores.

        (1) Compute mean average precision for all queries using the "average_precision" function.
        (2) Compute the mean of all average precision scores
        Return the mean average precision score

        reference: https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map
        https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2
        """
        # TODO Put your code here.
        ###########################################################################
        average_precisions = []

        for query_id, relevant_docs in self.test_query_id_to_relevant_doc_ids.items():
            candidate_docs = self.query_id_to_ranked_doc_ids[query_id]
            ap = self.average_precision(relevant_docs, candidate_docs)
            average_precisions.append(ap)

        return np.mean(average_precisions)
        ###########################################################################

    def mean_average_precision_validation(self) -> float:
        """
        # Inputs:
            - None (uses ranked documents stored in self.query_id_to_ranked_doc_ids).

        # Output:
            - float: The MAP score, computed as the mean of all average precision scores.

        (1) Compute mean average precision for all queries using the "average_precision" function.
        (2) Compute the mean of all average precision scores
        Return the mean average precision score

        reference: https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map
        https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2
        """
        # TODO Put your code here.
        ###########################################################################
        average_precisions = []

        for query_id, relevant_docs in self.validation_query_id_to_relevant_doc_ids.items():
            candidate_docs = self.query_id_to_ranked_doc_ids[query_id]
            ap = self.average_precision(relevant_docs, candidate_docs)
            average_precisions.append(ap)

        return np.mean(average_precisions)

    # Task 4: Ranking the Top 10 Documents based on Similarity Scores (10 Pts)

    def show_ranking_documents(self, example_query: str) -> None:

        """
        # Inputs:
            - example_query (str): A query string for which top-ranked documents should be displayed.

        # Output:
            - None (prints the ranked documents along with similarity scores).

        (1) rank documents with given query with cosine similarity scores
        (2) prints the top 10 results along with its similarity score.

        """
        # TODO Put your code here.
        query_embedding = self.model.encode(example_query)
        document_embeddings = self.model.encode(self.documents)
        ###########################################################################
        similarities = cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)
        ranked_doc_indices = np.argsort(similarities[0])[::-1][:self.top_k]

        print("Top 10 documents for the query '{}':".format(example_query))
        for i, index in enumerate(ranked_doc_indices):
            document_id = self.document_ids[index]
            similarity_score = similarities[0][index]
            print("Rank {}: Document ID: {}, Similarity Score: {}".format(i + 1, document_id, similarity_score))
        ###########################################################################

    # Task 5:Fine tune the sentence transformer model (25 Pts)
    # Students are not graded on achieving a high MAP score.
    # The key is to show understanding, experimentation, and thoughtful analysis.

    def fine_tune_model(self, batch_size: int = 32, num_epochs: int = 3,
                        save_model_path: str = "finetuned_senBERT"):

        """
        Fine-tunes the model using MultipleNegativesRankingLoss.
        (1) Prepare training examples from `self.prepare_training_examples()`
        (2) Experiment with [anchor, positive] vs [anchor, positive, negative]
        (3) Define a loss function (`MultipleNegativesRankingLoss`)
        (4) Freeze all model layers except the final layers
        (5) Train the model with the specified learning rate
        (6) Save the fine-tuned model
        """
        # TODO Put your code here.
        ###########################################################################
        

        device = torch.device("cuda")

        train_examples = self.prepare_training_examples()
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        self.model.to(device)

        for param in self.model[0].auto_model.parameters():
            param.requires_grad = False

        for param in self.model[0].auto_model.encoder.layer[-1].parameters():
            param.requires_grad = True

        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs,
                       warmup_steps=int(0.1 * len(train_dataloader)), show_progress_bar=True,
                       output_path=save_model_path)

        ###########################################################################
        self.rank_documents(encoding_method='sentence_transformer')
        map_score = self.mean_average_precision()
        map_score_validation = self.mean_average_precision_validation()

        return map_score, map_score_validation

    # Take a careful look into how the training set is created
    def prepare_training_examples(self) -> list[InputExample]:

        """
        Prepares training examples from the training data.
        # Inputs:
            - None (uses self.train_query_id_to_relevant_doc_ids to create training pairs).

         # Output:
            Output: - list[InputExample]: A list of training samples containing [anchor, positive] or [anchor, positive, negative].

        """
        train_examples = []

        all_doc_ids = set(self.document_ids)
        for qid, doc_ids in self.train_query_id_to_relevant_doc_ids.items():
            anchor = self.query_id_to_text[qid]
            relevant_doc_ids = set(doc_ids)
            for doc_id in doc_ids:
                positive = self.document_id_to_text[doc_id]

                negative_doc_ids = list(all_doc_ids - relevant_doc_ids)
                negative_id = random.choice(negative_doc_ids)
                negative = self.document_id_to_text[negative_id]

                train_examples.append(InputExample(texts=[anchor, positive, negative]))
                # train_examples.append(InputExample(texts=[anchor, positive]))

        return train_examples


# Initialize and use the model
model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels")

# Compare the outputs
# print("Ranking with sentence_transformer...")
# model.rank_documents(encoding_method='sentence_transformer')
# map_score = model.mean_average_precision()
# map_score_validation = model.mean_average_precision_validation()
# print("Mean Average Precision:", map_score)
# print("Mean Average Precision Validation:", map_score_validation)
#
# # Compare the outputs
# print("Ranking with glove...")
# model.rank_documents(encoding_method='glove')
# map_score = model.mean_average_precision()
# print("Mean Average Precision:", map_score)
#
#
# model.show_ranking_documents("Breast Cancer Cells Feed on Cholesterol")

# Finetune all-MiniLM-L6-v2 sentence transformer model
map_score, map_score_validation= model.fine_tune_model(batch_size=32, num_epochs=20,
                                  save_model_path="epoch_20_no_full_layer_negetive_finetuned_senBERT_train_v2")  # Adjust batch size and epochs as needed

print("Mean Average Precision:", map_score)
print("Mean Average Precision Validation:", map_score_validation)
