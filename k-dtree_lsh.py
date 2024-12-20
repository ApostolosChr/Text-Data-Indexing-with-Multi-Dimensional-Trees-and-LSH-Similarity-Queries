import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np
import nltk
from sklearn.neighbors import KDTree
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords   #3ο Να βάλω δίαστημα στο dblp 
nltk.download('punkt') 
nltk.download('stopwords')  # Κατέβασε τα δεδομένα για τις stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Συνάρτηση για την ανάκτηση των πληροφοριών εκπαίδευσης
def get_education_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Αναζήτηση εκπαιδευτικής ενότητας με βάση τις συνώνυμες λέξεις
    education_section = soup.find(re.compile(r'(h[1-6]|p|div|section)', re.IGNORECASE), string=re.compile(r'(education|training|learning|instruction|biography)', re.IGNORECASE))

    if education_section:
        # Αν βρέθηκε η εκπαιδευτική ενότητα, προχωρήστε στην ανάκτηση των πληροφοριών
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


def calculate_text_similarity(education_texts, user_query, threshold):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(education_texts + [user_query])
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    max_similarity_indexes = np.where(similarity_scores >= threshold)[0]

    scientist_education_mapping = {}
    for idx in max_similarity_indexes:
        scientist_education_mapping[filtered_scientists[idx]] = (similarity_scores[idx], education_texts[idx])

    return scientist_education_mapping, similarity_scores


def filter_scientists(combined_data, kdt, min_awards, min_dblp, initial_letters):
    initial_range = initial_letters.upper()
    filtered_indices = []

    for letter in range(ord(initial_range[0]), ord(initial_range[-1]) + 1):
        sample = np.array([[min_awards, min_dblp, letter - 64]])
        
        closest_indices = kdt.query_radius(sample, r=np.inf)[0]
        filtered_indices.extend([index for index in closest_indices if
                                 combined_data[index][0] >= min_awards and
                                 combined_data[index][1] >= min_dblp and
                                 combined_data[index][2] == letter - 64])

    filtered_scientists = [data[index][0] for index in filtered_indices]
    return filtered_scientists

 


def extract_names(url): 
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    scientists_by_initial = {}  # Δημιουργία λεξικού για τους επιστήμονες ανά αρχικό γράμμα
    letter_codes = {chr(i): i - ord('A') + 1 for i in range(ord('A'), ord('Z') + 1)}

    # Αναζήτηση του στοιχείου h2 με id="A"
    start_tag = soup.find('span', {'id': 'A'}).parent.parent

    # Αναζήτηση του στοιχείου h2 με id="See_also"
    end_tag = soup.find('span', {'id': 'See_also'}).parent

    # Βρίσκουμε τα επιστημονικά ονόματα από το "A" μέχρι το "Z"
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

                # Αντιστοίχιση αριθμών με τα αρχικά
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
    
        if scientist_count >= 100:
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
            initial_codes = extract_names(initial_url)  # Παίρνουμε τα αρχικά κώδικα από τη συνάρτηση extract_names(url)
            initial_code = None

            # Βρίσκουμε το initial_code που αντιστοιχεί στον επιστήμονα
            for code, scientists_list in initial_codes.items():
                if name in scientists_list:
                    initial_code = int(code)
                    break
            
            # Αν δε βρεθεί ο κωδικός, βάζουμε 0
            if initial_code is None:
                initial_code = 0
            
            # Προσθέτουμε το initial_code στη λίστα codes_list
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

print("Combined Data:")
print(combined_data)

kdt = KDTree(combined_data, leaf_size=30, metric='euclidean')

while True:
    initial_letters = input("Δώστε ένα διάστημα αρχικών γραμμάτων στα λατινικά (π.χ., 'A-D'): ").upper()
    
    if initial_letters.lower() == 'exit':
        break
    
    while not re.match(r'^[A-Z]-[A-Z]$', initial_letters):
        print("Το διάστημα πρέπει να έχει τη μορφή 'A-D' στα λατινικά.")
        initial_letters = input("Δώστε ένα διάστημα αρχικών γραμμάτων (π.χ., 'A-D'): ").upper()

    min_awards = int(input("Δώστε το ελάχιστο αριθμό βραβείων: "))
    min_dblp = int(input("Δώστε το ελάχιστο αριθμό δημοσιεύσεων στο DBLP: "))
    
    # Εκτύπωση ονομάτων επιστημόνων που πληρούν τα κριτήρια
    filtered_scientists = filter_scientists(combined_data, kdt, min_awards, min_dblp, initial_letters)
    if not filtered_scientists:
        print("Δεν βρήκαμε επιστήμονα με αυτά τα κριτήρια.")
        continue
    else:
        print("Οι επιστήμονες που πληρούν τα κριτήρια είναι:")
        print(filtered_scientists)

    # Εισαγωγή του ερωτήματος από τον χρήστη
    user_query = input("Δώστε το ερώτημα που θέλετε να αναζητήσετε σχετικά με την εκπέδευση των επιστημόνων: ")
    
    # Φιλτράρισμα εκπαιδευτικών κειμένων και υπολογισμός ομοιότητας
    filtered_education_texts = []
    for scientist in filtered_scientists:
        education_texts = [edu_info for name, edu_info in names_education if name == scientist]
        filtered_education_texts.extend(education_texts)

    threshold = 0.025  # Ορίστε το όριο ομοιότητας
    result, similarity_scores = calculate_text_similarity(filtered_education_texts, user_query, threshold=threshold)

    if not result:
        print("Δεν βρέθηκαν αποτελέσματα πάνω από το όριο ομοιότητας.")
    else:
        print("Επιστήμονες με ποσοστό ομοιότητας πάνω από το όριο:")
        for scientist, (similarity, education) in result.items():
            print(f"Επιστήμονας: {scientist}")
            print(f"Ποσοστό ομοιότητας: {similarity}")
            print(f"Κείμενο εκπαίδευσης:\n{education}\n")