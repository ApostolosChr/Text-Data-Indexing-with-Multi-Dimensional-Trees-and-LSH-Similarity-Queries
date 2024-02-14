import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np
import nltk
from rtree import index
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords   
nltk.download('punkt') 
nltk.download('stopwords')  # Κατέβασε τα δεδομένα για τις stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    # Δημιουργία του TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Μετατροπή των κειμένων σε TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(education_texts + [user_query])

    # Υπολογισμός της ομοιότητας μεταξύ του ερωτήματος του χρήστη και των κειμένων εκπαίδευσης
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    max_similarity_indexes = np.where(similarity_scores >= threshold)[0]

    scientist_education_mapping = {}
    for idx in max_similarity_indexes:
        scientist_education_mapping[filtered_scientists_final[idx]] = (similarity_scores[idx], education_texts[idx])

    return scientist_education_mapping, similarity_scores

def filter_by_rtree(rtree_index, min_awards, min_dblp):
    filtered_scientists_names = []

    overlap_bbox = (min_awards, min_dblp, max_awards, max_dblp)

    # Use intersection query to get the matching scientists
    for idx in rtree_index.intersection(overlap_bbox):
        filtered_scientists_names.append(data[idx][0])

    return filtered_scientists_names

def filter_by_initial_letters(combined_data, initial_letters):
    initial_range = initial_letters.upper()
    filtered_scientists_initial = []

    for letter in range(ord(initial_range[0]), ord(initial_range[-1]) + 1):
        # combined_data: [awards, dblp, initial_letter_code]
        scientists_with_letter = [data for data in combined_data if data[2] == float(letter - 64)]
        filtered_scientists_initial.extend(scientists_with_letter)

    return filtered_scientists_initial

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

scientist_name = []

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
        education_section = soup.find('span', {'id': re.compile(r'.*ducation.*')})
        
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


            scientist_name.append([name])

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


print("Combined Data:\n", combined_data)

max_awards = np.nanmax(awards_dblp_data[:, 0])
max_dblp = np.nanmax(awards_dblp_data[:, 1])

rtree_index = index.Index()
for idx, scientist_data in enumerate(data):
    awards, dblp, _ = scientist_data[1], scientist_data[2], idx
    
    # Εάν οι τιμές είναι None, θέτουμε σε 0
    if awards is None:
        awards = 0
    if dblp is None:
        dblp = 0
    
    bounding_box = (awards, dblp, awards, dblp)  # Ορισμός ορθογωνίου επικάλυψης
    rtree_index.insert(idx, bounding_box)  # Εισαγωγή στο R-tree index

# Loop για το user input
while True:
    initial_letters = input("\nΔώσε ένα διάστημα αρχικών γραμμάτων (π.χ., 'A-D'): ")
    
    if initial_letters.lower() == 'exit':
        break

    # First of all, filter the scientists by the given initial letters:
    filtered_scientists_initial = filter_by_initial_letters(combined_data, initial_letters)

    min_awards = int(input("Δώσε το ελάχιστο αριθμό βραβείων: "))
    min_dblp = int(input("Δώσε το ελάχιστο αριθμό δημοσιεύσεων στο DBLP: "))

    # Then, filter the scientists by the given min_awards and min_dblp:
    filtered_scientists_final = filter_by_rtree(rtree_index, min_awards, min_dblp)
    print("The filtered scientists are: ", filtered_scientists_final)


    filtered_education_texts = []
    for scientist in filtered_scientists_final:
        education_texts = [edu_info for name, edu_info in names_education if name == scientist]
        filtered_education_texts.extend(education_texts)

    user_query = input("\nΔώσε το ερώτημα που θέλεις να αναζητήσεις στο κείμενο εκπαίδευσης: ")

    threshold = 0.005  # Ορίστε το όριο ομοιότητας
    result, similarity_scores = calculate_text_similarity(filtered_education_texts, user_query, threshold=threshold)

    if not result:
        print("Δεν βρέθηκαν αποτελέσματα πάνω από το όριο ομοιότητας.")
    else:
        print("Επιστήμονες με ποσοστό ομοιότητας πάνω από το όριο:")
        for scientist, (similarity, education) in result.items():
            print(f"Επιστήμονας: {scientist}")
            print(f"Ποσοστό ομοιότητας: {similarity}")
            print(f"Κείμενο εκπαίδευσης:\n{education}\n")