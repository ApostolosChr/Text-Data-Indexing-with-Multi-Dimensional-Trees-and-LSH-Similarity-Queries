import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
nltk.download('punkt')
nltk.download('stopwords')

# Συνάρτηση για την ανάκτηση των πληροφοριών εκπαίδευσης
def get_education_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    education_section = soup.find(re.compile(r'(h[1-6]|p|div|section)', re.IGNORECASE), string=re.compile(r'(education|training|learning|instruction|biography)', re.IGNORECASE))

    if education_section:
        try:
            education_info = re.sub(r'\[\D*\d+\]', '', education_section.find_next('p').get_text())
            return education_info.strip()
        except AttributeError:
            return None

    return None

# Συνάρτηση για την ανάκτηση των πληροφοριών βραβείων
def get_awards_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    honors_awards_section = soup.find(re.compile(r'(span|h[1-6]|p|div|section)', re.IGNORECASE), string=re.compile(r'(honors|awards|Achievements|Recognition|Prizes)', re.IGNORECASE))

    if honors_awards_section:
        awards_list = honors_awards_section.find_next('ul')
        if awards_list:
            awards_count = len(awards_list.find_all('li'))
            return awards_count

    return None


def get_dblp_publications(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    matches_element = soup.find("p", id="completesearch-info-matches")
    if matches_element:
        matches_text = matches_element.text.split()[1].replace(",", "")
        try:
            matches = int(matches_text)
            return matches
        except ValueError:
            pass
#Δημιουργια κλάσης Node
class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right
#Δημιουργια κλάσης RangeTree3D
class RangeTree3D:
    def __init__(self):
        self.root = None

    def insert(self, point):
        if self.root is None:
            self.root = Node(point)
        else:
            self._insert_recursive(self.root, point, 0)

    def _insert_recursive(self, node, point, depth):
        axis = depth % 3

        if point[axis] < node.point[axis]:
            if node.left is None:
                node.left = Node(point)
                print(f"Inserted {point} as left child of {node.point}")
            else:
                self._insert_recursive(node.left, point, depth + 1)
        else:
            if node.right is None:
                node.right = Node(point)
                print(f"Inserted {point} as right child of {node.point}")
            else:
                self._insert_recursive(node.right, point, depth + 1)

    def search(self, min_point, max_point):
        result = []
        print("Initial Range:", min_point, "-", max_point)
        self._search_recursive(self.root, min_point, max_point, 0, result)
        return result

    def _search_recursive(self, node, min_point, max_point, depth, result):
        if node is None:
            return

        axis = depth % 3
        
        if min_point[axis] <= node.point[axis] <= max_point[axis]:
            if all(min_point[d] <= node.point[d] <= max_point[d] for d in range(3)):
                result.append(node.point)

            if node.left is not None:
                self._search_recursive(node.left, min_point, max_point, depth + 1, result)

            if node.right is not None and max_point[axis] >= node.point[axis]:
                self._search_recursive(node.right, min_point, max_point, depth + 1, result)
        elif min_point[axis] > node.point[axis]:
            self._search_recursive(node.right, min_point, max_point, depth + 1, result)
        else:
            self._search_recursive(node.left, min_point, max_point, depth + 1, result)



def highlight_common_words(text1, text2):
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    common_words = words1.intersection(words2)

    for word in common_words:
        text1 = re.sub(r'\b' + re.escape(word) + r'\b', f"\033[4m{word}\033[0m", text1, flags=re.IGNORECASE)
        text2 = re.sub(r'\b' + re.escape(word) + r'\b', f"\033[4m{word}\033[0m", text2, flags=re.IGNORECASE)

    return text1, text2


def calculate_text_similarity(education_texts, threshold_percentage):
  
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(education_texts)
    
    
    minhash_list = []
    for text in education_texts:
        tokens = text.split()  
        minhash = MinHash()
        for token in tokens:
            minhash.update(token.encode('utf-8'))
        minhash_list.append(minhash)

    
    lsh = MinHashLSH(threshold=threshold_percentage / 100, num_perm=128)
    for i, minhash in enumerate(minhash_list):
        lsh.insert(i, minhash)
    
   
    candidate_pairs = set()
    for i, minhash in enumerate(minhash_list):
        results = lsh.query(minhash)
        for j in results:
            if i != j:  
                candidate_pairs.add(tuple(sorted([i, j])))
    
    similarity_mapping = defaultdict(list)
    for pair in candidate_pairs:
        i, j = pair
        similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
        similarity_mapping[i].append((j, similarity))
    
    return similarity_mapping


#Τροποποίηση της μεθόδου filter_sientists
def filter_scientists(combined_data, range_tree, min_awards, min_dblp, max_user_dblp, initial_letters):
    initial_range = initial_letters.upper()
    filtered_scientists = []

    # Μετατροπή initial letters σε αντίστοιχα indices
    initial_indices = [ord(char) - ord('A') + 1 for char in initial_range if 'A' <= char <= 'Z']

    # Βρισκει maximum values για awards και DBLP
    max_awards = max(combined_data, key=lambda x: x[0])[0]
    max_dblp = max_user_dblp

    for initial_index in range(initial_indices[0], initial_indices[-1] + 1):
        # Δημιουργει  query point
        query_point = [min_awards, min_dblp, initial_index]

        
        search_upper_bound = [max_awards, max_dblp, initial_index]

       
        search_result = range_tree.search(query_point, search_upper_bound)

       
        for point in search_result:
        # Εύρεση της γραμμής που αντιστοιχεί στο σημείο point
            row_index = None
            for i, row in enumerate(combined_data):
                if np.array_equal(row, point):
                    row_index = i
                    break
            if row_index is not None:
            # Προσθήκη της γραμμής στη λίστα filtered_scientists
             filtered_scientists.append(data[row_index][0])
    return filtered_scientists


def extract_names(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    scientists_by_initial = {}
    letter_codes = {chr(i): i - ord('A') + 1 for i in range(ord('A'), ord('Z') + 1)}

    start_tag = soup.find('span', {'id': 'A'}).parent.parent
    end_tag = soup.find('span', {'id': 'See_also'}).parent

    for tag in start_tag.find_all_next():
        if tag == end_tag:
            break
        if tag.name == 'h2':
            initial_text = tag.text.strip().split('[')[0]
            initial = initial_text.split()[0]
            if initial >= 'A' and initial <= 'Z':
                scientists_list = []
                scientists = tag.find_next('ul').find_all('li')

                for scientist in scientists:
                    name = scientist.text.split(' – ')[0]
                    scientists_list.append(name)

                initial_code = str(letter_codes[initial])
                scientists_by_initial[initial_code] = scientists_list
      
    return scientists_by_initial

# Αρχικό URL
initial_url = 'https://en.wikipedia.org/wiki/List_of_computer_scientists'

names_education = [] 
codes_list = []
names_list = extract_names(initial_url)
data = []

scientist_count = 0

for initial, scientists in names_list.items():
    for name in scientists:
        if scientist_count >= 4:
            break
            
        formatted_name = re.sub(r'\s+', '_', name.strip())
        scientist_url = f'https://en.wikipedia.org/wiki/{formatted_name}'
        dblp_url =  f'https://dblp.org/search/publ?q=author:{formatted_name}'
        
        response = requests.get(scientist_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        education_section = soup.find(re.compile(r'(h[1-6]|p|div|section)', re.IGNORECASE), string=re.compile(r'(education|training|learning|instruction|biography)', re.IGNORECASE))

        if education_section:
            try:
                education_info = education_section.find_next('p').get_text()
                names_education.append((name, education_info.strip()))

                
                
            except AttributeError:
                pass

        
        education_info = get_education_info(scientist_url)
        awards_count = get_awards_info(scientist_url)
        dblp_publications = get_dblp_publications(dblp_url)

        if education_info:
            data.append([name, awards_count, dblp_publications])
           
            print(f"Name: {name}")
            print(f"Education: {education_info}")
            if awards_count:
                print(f"Awards: {awards_count}")
            else:
                print("Awards: No information available")
            if dblp_publications is not None:
                print(f"DBLP: {dblp_publications}\n")
            else:
                print("DBLP: No information available\n")
            print("------------")
            initial_codes = extract_names(initial_url)  
            initial_code = None

            
            for code, scientists_list in initial_codes.items():
                if name in scientists_list:
                    initial_code = int(code)
                    break
            
            if initial_code is None:
                initial_code = 0
            
            
            codes_list.append(initial_code)

        scientist_count += 1
        time.sleep(1)

codes_array = np.array(codes_list).reshape(-1, 1)
data_array = np.array(data)

awards_dblp_data = data_array[:, 1:3].astype(float)
combined_data = np.concatenate((awards_dblp_data, codes_array), axis=1)
combined_data = np.nan_to_num(combined_data, nan=0, posinf=0, neginf=0)
combined_data = combined_data.astype(float)
combined_data[np.isnan(combined_data)] = 0 

#sxolio dikom
print("Combined Data:")
print(combined_data)
#sxolio dikom
range_tree = RangeTree3D()
for data_point in combined_data:
    print("Inserting data point into range tree:", data_point)
    range_tree.insert(data_point)

print("Range tree constructed successfully.")

while True:
    initial_letters = input("Δώστε ένα διάστημα αρχικών γραμμάτων στα λατινικά (π.χ.    , 'A-D'): ").upper()
    
    if initial_letters.lower() == 'exit':
        break
    
    while not re.match(r'^[A-Z]-[A-Z]$', initial_letters):
        print("Το διάστημα πρέπει να έχει τη μορφή 'A-D' στα λατινικά.")
        initial_letters = input("Δώστε ένα διάστημα αρχικών γραμμάτων (π.χ., 'A-D'): ").upper()

    min_awards = int(input("Δώστε το ελάχιστο αριθμό βραβείων: "))
    while True:
        input_dblp = input("Δώσε ένα εύρος τιμών για τον αριθμό δημοσιέυσεων στο DBLP (π.χ. 65 - 3456): ")

        # Check if the input matches the desired pattern for a range of numbers
        match = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', input_dblp)

        if match:
            min_str, max_str = match.groups()

            # Convert the strings to integers
            min_dblp = int(min_str)
            max_user_dblp = int(max_str)
            break
        else:
            print("Το διάστημα πρέπει να είναι της μορφής 'Integer - Integer'. Παρακαλώ δοκιμάστε ξανά.")
    
    filtered_scientists = filter_scientists(combined_data, range_tree, min_awards, min_dblp, max_user_dblp, initial_letters)
    print("Filtered Scientists:", filtered_scientists)
    if not filtered_scientists:
        print("Δεν βρήκαμε επιστήμονα με αυτά τα κριτήρια.")
        continue
    else:
        print("Οι επιστήμονες που πληρούν τα κριτήρια είναι:")
        print(filtered_scientists)

    filtered_education_texts = [edu_info for name, edu_info in names_education if any(name in scientist for scientist in filtered_scientists)]


    threshold = float(input("Εισαγάγετε το ποσοστό ομοιότητας που επιθυμείτε (ανάμεσα σε 0 και 1): "))
    result = calculate_text_similarity(filtered_education_texts, threshold)
    
    if not result:
        print("Δεν βρέθηκαν αποτελέσματα πάνω από το όριο ομοιότητας.")
        continue
   
    for i, similar_texts in result.items():
        scientist1 = filtered_scientists[i]
        matching_educations = [edu_info for name, edu_info in names_education if name == scientist1]
        if matching_educations:
            education1 = matching_educations[0]
            education1 = re.sub(r'\[\D*\d+\]', '', education1)
            for j, similarity in similar_texts:
                scientist2 = filtered_scientists[j]
                matching_educations2 = [edu_info for name, edu_info in names_education if name == scientist2]
                if matching_educations2:
                    education2 = matching_educations2[0]
                    education2 = re.sub(r'\[\D*\d+\]', '', education2)
                    similarity_percentage = similarity * 100
                    highlighted_education1, highlighted_education2 = highlight_common_words(education1, education2)
                    if similarity_percentage >= threshold * 100:
                        print(f"Ζεύγη Επιστημόνων με ποσοστό ομοιότητας εκπέδευσης πάνω απο {threshold*100}%: {scientist1} με {scientist2}:")
                        print(f"  - Ποσοστό Ομοιότητας εκπέδευσης επιστημόνων: {similarity_percentage:.2f}%")
                        print(f"  - Κείμενο Εκπαίδευσης Επιστήμονα {scientist1}:\n{highlighted_education1}\n")
                        print(f"  - Κείμενο Εκπαίδευσης Επιστήμονα {scientist2}:\n{highlighted_education2}\n")
                        print("-------------------------------------------------------------------------------------\n")
    print("Δεν βρέθηκαν άλλα αποτελέσματα πάνω από το όριο ομοιότητας.")