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
import time
from rtree import index

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

# Συνάρτηση για την ανάκτηση των δημοσιεύσεων από την DBLP
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


def highlight_common_words(text1, text2):
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    common_words = words1.intersection(words2)

    for word in common_words:
        text1 = re.sub(r'\b' + re.escape(word) + r'\b', f"\033[4m{word}\033[0m", text1, flags=re.IGNORECASE)
        text2 = re.sub(r'\b' + re.escape(word) + r'\b', f"\033[4m{word}\033[0m", text2, flags=re.IGNORECASE)

    return text1, text2


def calculate_text_similarity(education_texts, threshold_percentage):
    # Δημιουργία του TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(education_texts)
    
    # Δημιουργία των MinHash signatures
    minhash_list = []
    for text in education_texts:
        tokens = text.split()  # Διαχωρισμός του κειμένου σε λέξεις
        minhash = MinHash()
        for token in tokens:
            minhash.update(token.encode('utf-8'))
        minhash_list.append(minhash)

    # Δημιουργία του LSH index
    lsh = MinHashLSH(threshold=threshold_percentage / 100, num_perm=128)
    for i, minhash in enumerate(minhash_list):
        lsh.insert(i, minhash)
    
    # Επιλογή των πιθανών ζευγαριών κειμένων που αξίζει να εξεταστούν
    candidate_pairs = set()
    for i, minhash in enumerate(minhash_list):
        results = lsh.query(minhash)
        for j in results:
            if i != j:  # Αποφυγή σύγκρισης κειμένων με τον εαυτό τους
                candidate_pairs.add(tuple(sorted([i, j])))
    
    # Υπολογισμός της πραγματικής ομοιότητας για τα πιθανά ζευγάρια
    similarity_mapping = defaultdict(list)
    for pair in candidate_pairs:
        i, j = pair
        similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
        similarity_mapping[i].append((j, similarity))
    
    return similarity_mapping


# Συνάρτηση για το φιλτράρισμα των επιστημόνων βάσει κριτηρίων
def filter_scientists(combined_data, rtree, min_awards, min_dblp, max_user_dblp, initial_letters):
    initial_range = initial_letters.upper()
    filtered_indices = []

    # Υπολογισμός του max_x και max_y
    max_x = max([x for x, _, _ in combined_data]) + 1
    max_y = max_user_dblp

    for letter in range(ord(initial_range[0]), ord(initial_range[-1]) + 1):
        min_x = min_awards
        min_y = min_dblp
        # Υπολογισμός του κωδικού για το γράμμα (A=1, B=2, κλπ)
        letter_code = letter - ord('A') + 1
        min_z = letter_code
        max_z = letter_code + 1
        
        bbox = (min_x, min_y, min_z, max_x, max_y, max_z)
        closest_indices = rtree.intersection(bbox)
        
        # Έλεγχος κάθε σημείου αν είναι εντός του παραλληλεπιπέδου και πληροί τα κριτήρια
        for index in closest_indices:
            x, y, z = combined_data[index]
            if x >= min_awards and y >= min_dblp and min_z <= z < max_z:
                filtered_indices.append(index)

    filtered_scientists = [data[index][0] for index in filtered_indices]
    return filtered_scientists


# Συνάρτηση για την ανάκτηση των ονομάτων των επιστημόνων ανά αρχικό γράμμα
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
        if scientist_count >= 3:
            break
            
        formatted_name = re.sub(r'\s+', '_', name.strip())
        scientist_url = f'https://en.wikipedia.org/wiki/{formatted_name}'
        dblp_url =  f'https://dblp.org/search/publ?q=author:{formatted_name}'
        
        response = requests.get(scientist_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        education_section = soup.find(re.compile(r'(h[1-6]|p|div|section)', re.IGNORECASE), string=re.compile(r'(education|training|learning|instruction|biography)', re.IGNORECASE))
        
        education_info = get_education_info(scientist_url)
        awards_count = get_awards_info(scientist_url)
        dblp_publications = get_dblp_publications(dblp_url)

        names_education.append((name, education_info.strip()))

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

p = index.Property()
p.dimension = 3
idx3d = index.Index(properties=p)

for i, row in enumerate(combined_data):
    x, y, z = row
    idx3d.insert(i, (x, y, z, x, y, z))  # Μετατροπή των συντεταγμένων σε μορφή (minx, miny, minz, maxx, maxy, maxz)

# Δημιουργία ενός παραλληλεπιπέδου για την αναζήτηση
min_x = np.min(combined_data[:, 0])
max_x = np.max(combined_data[:, 0])
min_y = np.min(combined_data[:, 1])
max_y = np.max(combined_data[:, 1])
min_z = np.min(combined_data[:, 2])
max_z = np.max(combined_data[:, 2])

query_bbox = (min_x, min_y, min_z, max_x, max_y, max_z)
results = list(idx3d.intersection(query_bbox))

while True:
    initial_letters = input("Δώστε ένα διάστημα αρχικών γραμμάτων στα λατινικά (π.χ., 'A-D'): ").upper()
    
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
    
    start_time = time.time()

    filtered_scientists = filter_scientists(combined_data, idx3d, min_awards, min_dblp, max_user_dblp, initial_letters)

    filtering_time = time.time() - start_time
    print("Ο χρόνος που απαιτείται για το φιλτράρισμα με χρήση του RTree:", filtering_time, "seconds")
    
    if not filtered_scientists:
        print("Δεν βρήκαμε επιστήμονα με αυτά τα κριτήρια.")
        continue
    else:
        print("Οι επιστήμονες που πληρούν τα κριτήρια είναι:")
        print(filtered_scientists)

    filtered_education_texts = [edu_info for name, edu_info in names_education if name in filtered_scientists]

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