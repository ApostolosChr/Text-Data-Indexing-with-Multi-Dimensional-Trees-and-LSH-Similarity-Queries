import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np
import nltk
from sklearn.neighbors import KDTree
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
nltk.download('punkt') 
nltk.download('stopwords')  # Κατέβασε τα δεδομένα για τις stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Συνάρτηση για την ανάκτηση των πληροφοριών εκπαίδευσης
def get_education_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    education_section = soup.find('span', {'id': re.compile(r'.*ducation.*')})

    if education_section:
        try:
            education_info = education_section.find_next('p').get_text()
            return education_info.strip()
        except AttributeError:
            return None

    return None

# Συνάρτηση για την ανάκτηση των πληροφοριών βραβείων
def get_awards_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    honors_awards_section = soup.find('span', {'id': re.compile(r'.*Award.*')}) #πρέπει να βρώ πώς θα τα αναζητίσω τα awards 

    if honors_awards_section:
        awards_list = honors_awards_section.find_next('ul')
        if awards_list:
            awards_count = len(awards_list.find_all('li'))
            return awards_count

    return None

# Συνάρτηση για την εξαγωγή των ονομάτων
def extract_names(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    scientist_list = soup.find_all('li')

    names = []
    for scientist_section in scientist_list:
        anchor_tag = scientist_section.find('a')
        if anchor_tag:
            name = anchor_tag.get_text()
            names.append(name)

    return names

def get_dblp_publications(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    publications_section = soup.find('span', {'id': re.compile(r'.*publications.*')})

    if publications_section:
        publications_list = publications_section.find_next('ul')
        if publications_list:
            publications_count = len(publications_list.find_all('li'))
            return publications_count

    return None

def calculate_text_similarity(education_texts, user_query):
    # Δημιουργία του TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Μετατροπή των κειμένων σε TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(education_texts + [user_query])

    # Υπολογισμός της ομοιότητας μεταξύ του ερωτήματος του χρήστη και των κειμένων εκπαίδευσης
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    max_similarity_index = np.argmax(similarity_scores)
    max_similarity = similarity_scores[max_similarity_index]
    max_similar_scientist = filtered_scientists[max_similarity_index]
    max_similar_education = education_texts[max_similarity_index]

    # Επιστροφή του επιστήμονα και του κειμένου εκπαίδευσης με το μέγιστο ποσοστό ομοιότητας
    return max_similar_scientist, max_similar_education, max_similarity


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


# Αρχικό URL
initial_url = 'https://en.wikipedia.org/wiki/List_of_computer_scientists'
names = extract_names(initial_url)
names_education = [] 
data=[]

scientist_count = 0

# Για κάθε όνομα
for name in names:
    
    if scientist_count >= 100:
        break  # Σταματάει τον έλεγχο μετά τους πρώτους έξι επιστήμονες

    # Φτιάχνουμε το URL για τον κάθε επιστήμονα
    formatted_name = re.sub(r'\s+', '_', name.strip())  # Αντικαθίσταται το κενό με _
    scientist_url = f'https://en.wikipedia.org/wiki/{formatted_name}'
    response = requests.get(scientist_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    education_section = soup.find('span', {'id': re.compile(r'.*ducation.*')})
    
    if education_section:
        try:
            education_info = education_section.find_next('p').get_text()
            names_education.append((name, education_info.strip()))
        except AttributeError:
            pass
    
    # Παίρνουμε τις πληροφορίες για την εκπαίδευση
    education_info = get_education_info(scientist_url)

    # Παίρνουμε τον αριθμό των βραβείων
    awards_count = get_awards_info(scientist_url)

    # Παίρνουμε τις πληροφορίες για τις δημοσιεύσεις από το DBLP
    dblp_publications = get_dblp_publications(scientist_url)

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
        
    
    scientist_count += 1  # Αυξάνει τον μετρητή

    time.sleep(1)

data_array = np.array(data) # Μέχρι εδώ δουλεύει καλά 

awards_dblp_data = data_array[:, 1:3].astype(float)
initials_numbers = [ord(name[0].upper()) - 64 for name in data_array[:, 0]]
initials_array = np.array(initials_numbers).reshape(-1, 1)
combined_data = np.concatenate((awards_dblp_data, initials_array), axis=1) 

combined_data = np.nan_to_num(combined_data, nan=0, posinf=0, neginf=0)

print("Combined Data:")
print(combined_data)  # Εκτύπωση του combined_data για επιβεβαίωση

kdt = KDTree(combined_data, leaf_size=30, metric='euclidean')

while True:
    initial_letters = input("Δώσε ένα διάστημα αρχικών γραμμάτων (π.χ., 'A-D'): ")
    
    if initial_letters.lower() == 'exit':
        break

    min_awards = int(input("Δώσε το ελάχιστο αριθμό βραβείων: "))
    min_dblp = int(input("Δώσε το ελάχιστο αριθμό δημοσιεύσεων στο DBLP: "))
    user_query = input("Δώσε το ερώτημα που θέλεις να αναζητήσεις στο κείμενο εκπαίδευσης: ")
    filtered_scientists = filter_scientists(combined_data, kdt, min_awards, min_dblp, initial_letters)
    
    print(filtered_scientists)
    filtered_education_texts = []
    for scientist in filtered_scientists:
        education_texts = [edu_info for name, edu_info in names_education if name == scientist]
        filtered_education_texts.extend(education_texts)

    best_match_scientist, best_match_education, best_similarity = calculate_text_similarity(filtered_education_texts, user_query)

    print(f"Επιστήμονας με το υψηλότερο ποσοστό ομοιότητας: {best_match_scientist}")
    print(f"Κείμενο εκπαίδευσης: {best_match_education}")
    print(f"Ποσοστό ομοιότητας: {best_similarity}")